# Trueno: PyTorch/NumPy Replacement Specification

**Version**: 1.0
**Date**: 2025-11-17
**Status**: Living Document
**Priority**: CRITICAL - Core Strategic Positioning

---

## Executive Summary

Trueno is a high-performance compute library designed as a **drop-in replacement** for NumPy (data processing) and PyTorch (inference + training) in Rust-native applications. This specification defines the roadmap from current state (v0.2.0) to full replacement parity.

**Current State (v0.2.0)**:
- **NumPy Replacement**: ~35% complete (1D operations only)
- **PyTorch Replacement**: ~15% complete (inference only, no autograd)
- **Strategic Positioning**: High-performance 1D vector compute library with GPU acceleration

**Target State (v1.0.0)**:
- **NumPy Replacement**: 80%+ complete (multi-dimensional arrays, broadcasting, core ops)
- **PyTorch Replacement**: 60%+ complete (autograd, training, common layers)
- **Strategic Positioning**: Production-ready deep learning library for Rust

---

## 1. Current State Analysis (v0.2.0)

### 1.1 NumPy Replacement: ~35% Complete

#### ✅ What Works Today

**1D Element-wise Operations** (100% coverage):
```rust
use trueno::Vector;

// NumPy: np.add(a, b), np.exp(a), np.log(a)
let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
let result = a.add(&b).unwrap();  // Works identically to NumPy
```

**Supported operations**:
- Arithmetic: `add`, `sub`, `mul`, `div`, `neg`
- Transcendental: `exp`, `log`, `pow`, `sqrt`
- Trigonometric: `sin`, `cos`, `tan`
- Reductions: `sum`, `mean`, `std`, `min`, `max`, `norm`
- Linear algebra: `dot`, `matmul` (1D→2D via reshape)

**Reductions** (100% coverage for 1D):
```rust
// NumPy: np.sum(arr), np.mean(arr), np.std(arr)
let data = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
let total = data.sum().unwrap();      // 15.0
let average = data.mean().unwrap();   // 3.0
let stdev = data.std().unwrap();      // ~1.414
```

**GPU Acceleration** (14 operations):
- **OpComplexity::Low** (>100K threshold): `vec_add`, `dot`, `relu`, `leaky_relu`, `elu`, `sigmoid`, `tanh`, `swish`, `gelu`, `clip`
- **OpComplexity::Medium** (>10K threshold): `softmax`, `log_softmax`
- **OpComplexity::High** (>1K threshold): `matmul`, `convolve2d`

**Performance**:
- CPU SIMD: 2-8x faster than scalar (SSE2/AVX/AVX2)
- GPU: 10-50x faster than scalar for large workloads (>100K elements)

#### ❌ Critical Gaps (Blocking NumPy Replacement)

**Multi-dimensional Arrays** (0% coverage):
```python
# NumPy - WORKS
import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6]])  # 2D array (2×3)
arr.shape  # (2, 3)

# Trueno - MISSING
# Currently only supports 1D Vector<f32>
```

**Broadcasting** (0% coverage):
```python
# NumPy - WORKS
a = np.array([[1, 2, 3]])        # Shape (1, 3)
b = np.array([[1], [2], [3]])    # Shape (3, 1)
result = a + b                    # Shape (3, 3) via broadcasting

# Trueno - MISSING
# No broadcasting support, must manually tile/repeat
```

**Advanced Indexing** (0% coverage):
```python
# NumPy - WORKS
arr = np.array([1, 2, 3, 4, 5])
arr[1:4]        # Slicing: [2, 3, 4]
arr[[0, 2, 4]]  # Fancy indexing: [1, 3, 5]
arr[arr > 2]    # Boolean masking: [3, 4, 5]

# Trueno - MISSING
# Only supports full array operations
```

**Reshaping/Transposing** (0% coverage):
```python
# NumPy - WORKS
arr = np.array([1, 2, 3, 4, 5, 6])
arr.reshape(2, 3)   # [[1, 2, 3], [4, 5, 6]]
arr.transpose()     # Swap axes

# Trueno - MISSING
# No reshape or transpose operations
```

**Multiple Data Types** (0% coverage):
```python
# NumPy - WORKS
np.array([1, 2, 3], dtype=np.int32)
np.array([1.0, 2.0], dtype=np.float64)

# Trueno - ONLY f32
# Only supports Vector<f32>, no int32/float64/etc.
```

#### 🎯 Viable NumPy Use Cases Today

**1. 1D Signal Processing**:
```rust
use trueno::Vector;

// Audio processing: apply filter to 1D signal
let signal = Vector::from_slice(&audio_samples);
let filtered = signal.convolve2d(&filter_kernel).unwrap();
```

**2. Vector Similarity/Distance**:
```rust
// Compute cosine similarity between document embeddings
let doc1 = Vector::from_slice(&embedding1);
let doc2 = Vector::from_slice(&embedding2);
let similarity = doc1.dot(&doc2).unwrap() / (doc1.norm().unwrap() * doc2.norm().unwrap());
```

**3. Large-scale Element-wise Transformations** (GPU-accelerated):
```rust
// Activate 1M logits with GPU acceleration
let logits = Vector::from_slice(&vec![...; 1_000_000]);
let probs = logits.sigmoid().unwrap();  // Auto-uses GPU for >100K elements
```

---

### 1.2 PyTorch Replacement: ~15% Complete

#### ✅ What Works Today

**GPU-Accelerated Activations** (Inference Only):
```rust
use trueno::Vector;

// PyTorch: F.relu(x), F.sigmoid(x), F.gelu(x)
let hidden = Vector::from_slice(&[...]);
let activated = hidden.relu().unwrap();      // ReLU
let squashed = hidden.sigmoid().unwrap();    // Sigmoid
let gelu_out = hidden.gelu().unwrap();       // GELU (BERT/GPT)
let attention = scores.softmax().unwrap();   // Softmax (transformers)
```

**Supported activations** (14 GPU-accelerated):
- Standard: ReLU, Leaky ReLU, ELU, Sigmoid, Tanh
- Modern: Swish, GELU, Softmax, Log-Softmax
- Utility: Clip (gradient clipping)

**Basic Tensor Operations** (1D only):
```rust
// PyTorch: torch.dot(a, b), torch.matmul(A, B)
let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
let dot_product = a.dot(&b).unwrap();  // Works like PyTorch
```

**Performance** (GPU-accelerated):
- Element-wise ops (>100K): 10-50x faster than scalar
- Multi-pass ops (>10K): 5-20x faster than scalar (softmax, log_softmax)
- Matrix ops (>1K×1K): 10-50x faster than naive CPU

#### ❌ Critical Blockers (Preventing PyTorch Replacement)

**No Autograd** (0% coverage):
```python
# PyTorch - WORKS
import torch
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()       # Compute gradients
print(x.grad)      # [2.0, 4.0, 6.0]

# Trueno - MISSING
# Cannot train neural networks without gradient computation
```

**No Layers** (0% coverage):
```python
# PyTorch - WORKS
import torch.nn as nn
linear = nn.Linear(784, 128)
conv = nn.Conv2d(3, 64, kernel_size=3)
norm = nn.BatchNorm2d(64)

# Trueno - MISSING
# No nn::Module trait, no layer abstractions
```

**No Loss Functions** (0% coverage):
```python
# PyTorch - WORKS
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, targets)

# Trueno - MISSING
# Must manually implement loss computation
```

**No Optimizers** (0% coverage):
```python
# PyTorch - WORKS
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Trueno - MISSING
# No SGD, Adam, AdamW, etc.
```

**No Multi-dimensional Tensors** (0% coverage):
```python
# PyTorch - WORKS
x = torch.randn(32, 3, 224, 224)  # Batch of images (N, C, H, W)

# Trueno - MISSING
# Only 1D Vector<f32>
```

**No Model Saving/Loading** (0% coverage):
```python
# PyTorch - WORKS
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))

# Trueno - MISSING
# No checkpoint serialization
```

#### 🎯 Viable PyTorch Use Cases Today

**1. Neural Network Inference** (Forward Pass Only, 1D):
```rust
use trueno::Vector;

// Inference for a simple 1D feedforward network
let input = Vector::from_slice(&features);
let hidden1 = input.matmul(&weights1).unwrap().add(&bias1).unwrap().relu().unwrap();
let hidden2 = hidden1.matmul(&weights2).unwrap().add(&bias2).unwrap().gelu().unwrap();
let logits = hidden2.matmul(&weights3).unwrap().add(&bias3).unwrap();
let probs = logits.softmax().unwrap();
```

**2. Custom Activation Function Testing**:
```rust
// Compare custom activation vs standard activations
let data = Vector::from_slice(&test_data);
let relu_out = data.relu().unwrap();
let swish_out = data.swish().unwrap();
let gelu_out = data.gelu().unwrap();
```

**3. Batch Activation Processing** (GPU-accelerated):
```rust
// Process large batches of activations efficiently
let batch_logits = Vector::from_slice(&vec![...; 1_000_000]);  // 10K batch × 100 classes
let batch_probs = batch_logits.softmax().unwrap();  // GPU-accelerated
```

---

## 2. Strategic Gaps Analysis

### 2.1 Blocker Priorities (Must-Have for Replacement)

**Priority 1: Multi-dimensional Tensors** (Blocks 90% of use cases)
- **Blocker for**: Images, sequences, batches, convolutions, attention
- **Impact**: Cannot process 2D/3D/4D data (images, videos, point clouds)
- **Current workaround**: Flatten to 1D (loses structure, inefficient)
- **Target**: Phase 2 (v0.4.0 - v0.6.0)

**Priority 2: Autograd Engine** (Blocks training)
- **Blocker for**: Neural network training, gradient-based optimization
- **Impact**: Inference-only library (cannot train models)
- **Current workaround**: None (fundamental missing feature)
- **Target**: Phase 3 (v0.7.0)

**Priority 3: Broadcasting** (Blocks 70% of operations)
- **Blocker for**: Shape-polymorphic operations, batch processing
- **Impact**: Must manually tile/repeat tensors (inefficient, error-prone)
- **Current workaround**: Manual shape manipulation
- **Target**: Phase 2 (v0.5.0)

### 2.2 Nice-to-Have Features (Improve Usability)

**Multiple Data Types** (f16, f64, i32, etc.):
- **Impact**: Limited to f32 (no mixed precision, no integer ops)
- **Workaround**: Cast externally, use only f32
- **Target**: Phase 2 (v0.6.0)

**Advanced Indexing/Slicing**:
- **Impact**: Verbose data access, cannot use NumPy-style slicing
- **Workaround**: Manual index computation
- **Target**: Phase 2 (v0.5.0)

**Model Serialization**:
- **Impact**: Cannot save/load trained models
- **Workaround**: Manual weight export/import
- **Target**: Phase 3 (v1.0.0)

---

## 3. Phased Roadmap to Replacement Parity

### Phase 1: Complete 1D Operations (v0.2.x - v0.3.0)
**Timeline**: 2-3 months
**Goal**: Best-in-class 1D vector compute
**NumPy Parity**: ~40% (still 1D only)
**PyTorch Parity**: ~18% (inference only)

**Deliverables**:
- [x] GPU softmax/log_softmax (v0.2.1) ✅
- [ ] Remaining activations: hardswish, mish, selu
- [ ] GPU reductions: argmax/argmin, sum/mean/std
- [ ] GPU binary ops: add/sub/mul/div
- [ ] Async GPU API (batch operations)
- [ ] Comprehensive benchmarks (validate 10-50x claims)
- [ ] WASM SIMD128 backend

**Success Metrics**:
- ✅ All common 1D operations GPU-accelerated
- ✅ 10-50x GPU speedup validated by benchmarks
- ✅ Used in production by ≥3 projects

---

### Phase 2: Multi-Dimensional Tensors (v0.4.0 - v0.6.0)
**Timeline**: 6-12 months
**Goal**: NumPy-competitive for 2D/3D arrays
**NumPy Parity**: ~80% (multi-dim, broadcasting, core ops)
**PyTorch Parity**: ~20% (still no autograd)

#### v0.4.0: Tensor Type Foundation (3-4 months)

**Core Type**:
```rust
pub struct Tensor<T, const N: usize> {
    data: Vec<T>,
    shape: [usize; N],
    strides: [usize; N],
    offset: usize,
}

// Examples:
let scalar = Tensor::<f32, 0>::from_scalar(5.0);        // 0D scalar
let vector = Tensor::<f32, 1>::from_slice(&[1, 2, 3]);  // 1D vector
let matrix = Tensor::<f32, 2>::new([2, 3]);             // 2D matrix (2×3)
let image = Tensor::<f32, 3>::new([3, 224, 224]);       // 3D image (C, H, W)
let batch = Tensor::<f32, 4>::new([32, 3, 224, 224]);   // 4D batch (N, C, H, W)
```

**Storage**:
- Row-major layout (C-contiguous, matches NumPy default)
- Owned data vs. views (zero-copy slicing)
- Shape, strides, offset tracking for views

**2D Operations**:
```rust
// Transpose
let mat = Tensor::<f32, 2>::new([2, 3]);  // 2×3
let transposed = mat.transpose();         // 3×2

// Reshape
let vec = Tensor::<f32, 1>::new([6]);     // [1, 2, 3, 4, 5, 6]
let mat = vec.reshape([2, 3]);            // [[1, 2, 3], [4, 5, 6]]

// Flatten
let mat = Tensor::<f32, 2>::new([2, 3]);  // 2×3
let vec = mat.flatten();                  // 1D [6]

// Slicing
let mat = Tensor::<f32, 2>::new([5, 5]);
let sub = mat.slice([1..3, 2..4]);        // Sub-matrix (2×2 view)
```

**Row/Column Access**:
```rust
let mat = Tensor::<f32, 2>::new([3, 4]);
let row0 = mat.row(0);     // View of first row
let col1 = mat.column(1);  // View of second column
```

**2D Matrix Multiplication** (optimized):
```rust
// NumPy: C = A @ B
let a = Tensor::<f32, 2>::new([128, 256]);   // 128×256
let b = Tensor::<f32, 2>::new([256, 512]);   // 256×512
let c = a.matmul(&b).unwrap();               // 128×512 (GPU-accelerated)
```

**Deliverables**:
- Tensor<T, N> type with const generics
- Shape, strides, offset tracking
- Views vs owned data (zero-copy slicing)
- Transpose, reshape, flatten
- 2D matmul (GPU-accelerated, optimized)
- Row/column indexing

**Success Metrics**:
- Can represent images, batches, sequences
- Zero-copy views for efficient slicing
- 2D matmul competitive with NumPy

#### v0.5.0: Broadcasting (2-3 months)

**Broadcasting Rules** (NumPy-compatible):
```rust
// NumPy broadcasting semantics
let a = Tensor::<f32, 2>::new([3, 1]);   // Shape (3, 1)
let b = Tensor::<f32, 2>::new([1, 4]);   // Shape (1, 4)
let c = a.add(&b).unwrap();              // Shape (3, 4) via broadcasting

// Rules:
// 1. If ranks differ, prepend 1s to smaller rank
// 2. Dimensions are compatible if they're equal or one is 1
// 3. Broadcast stretches dimension 1 to match other dimension
```

**Shape Compatibility Checking**:
```rust
fn can_broadcast(shape_a: &[usize], shape_b: &[usize]) -> bool {
    // Implementation of NumPy broadcasting rules
}
```

**Element-wise Ops with Broadcasting**:
```rust
// All element-wise ops support broadcasting
let a = Tensor::<f32, 2>::new([64, 1]);     // Batch of 64 vectors
let bias = Tensor::<f32, 1>::new([128]);    // Bias vector
let result = a.add(&bias).unwrap();         // (64, 128) via broadcasting
```

**Advanced Indexing**:
```rust
// Boolean masking
let arr = Tensor::<f32, 1>::from_slice(&[1, 2, 3, 4, 5]);
let mask = arr.gt(2.0);                     // [false, false, true, true, true]
let filtered = arr.masked_select(&mask);    // [3, 4, 5]

// Integer array indexing
let indices = Tensor::<i32, 1>::from_slice(&[0, 2, 4]);
let selected = arr.index_select(&indices);  // [1, 3, 5]

// Slicing syntax (via macro)
let mat = Tensor::<f32, 2>::new([10, 10]);
let sub = tensor_slice!(mat, [1..5, ::2]);  // Rows 1-4, every other column
```

**Deliverables**:
- NumPy-compatible broadcasting rules
- Shape compatibility checking
- Broadcasted element-wise operations
- Boolean masking
- Integer array indexing
- Slicing macro (NumPy-style syntax)

**Success Metrics**:
- Pass 80%+ of NumPy broadcasting tests
- Idiomatic batch processing
- Efficient memory usage (avoid unnecessary copies)

#### v0.6.0: NumPy Parity (Core Ops) (3-4 months)

**All NumPy dtypes**:
```rust
pub enum DType {
    F16, F32, F64,
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    Bool,
}

let f32_tensor = Tensor::<f32, 2>::new([3, 3]);
let i32_tensor = Tensor::<i32, 2>::new([3, 3]);
let f64_tensor = Tensor::<f64, 2>::new([3, 3]);
```

**NumPy-style API**:
```rust
// Creation functions
let zeros = Tensor::<f32, 2>::zeros([3, 4]);
let ones = Tensor::<f32, 2>::ones([3, 4]);
let arange = Tensor::<f32, 1>::arange(0.0, 10.0, 1.0);
let linspace = Tensor::<f32, 1>::linspace(0.0, 1.0, 100);

// Concatenation/stacking
let a = Tensor::<f32, 2>::new([2, 3]);
let b = Tensor::<f32, 2>::new([2, 3]);
let cat = Tensor::concatenate(&[a, b], axis=0).unwrap();  // (4, 3)
let stack = Tensor::stack(&[a, b], axis=0).unwrap();      // (2, 2, 3)

// Splitting
let parts = Tensor::split(&tensor, 3, axis=0);  // Split into 3 parts

// Conditional operations
let mask = x.gt(0.5);
let result = Tensor::where(&mask, &x, &y);  // x if mask else y
let indices = Tensor::argwhere(&mask);      // Indices where mask is true
```

**Deliverables**:
- Generic dtype support (F16, F32, F64, I32, I64, U32, etc.)
- NumPy creation functions (zeros, ones, arange, linspace)
- Concatenate, stack, split
- Conditional operations (where, argwhere)
- Comprehensive test suite (80%+ NumPy compatibility)

**Performance Target**:
- 80-120% of NumPy speed (within 20% either way)

**Success Metrics**:
- Pass 80%+ of NumPy test suite (for covered ops)
- ≥10 production deployments
- Competitive performance with NumPy

---

### Phase 3: Autograd & Training (v0.7.0 - v1.0.0)
**Timeline**: 12-18 months
**Goal**: PyTorch-competitive for training
**NumPy Parity**: ~80% (stable)
**PyTorch Parity**: ~60% (training-ready)

#### v0.7.0: Autograd Engine (4-6 months)

**Computational Graph**:
```rust
pub struct Tensor<T, const N: usize> {
    data: TensorData<T, N>,
    grad: Option<Box<Tensor<T, N>>>,
    grad_fn: Option<Box<dyn BackwardFunction>>,
    requires_grad: bool,
}

// Example usage
let x = Tensor::<f32, 1>::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
let y = (&x * &x).sum();  // y = sum(x^2)
y.backward();             // Compute gradients
println!("{:?}", x.grad()); // Some([2.0, 4.0, 6.0])
```

**Reverse-mode Automatic Differentiation**:
- Dynamic graph construction (like PyTorch eager mode)
- Gradient tape for backpropagation
- Efficient memory management (release intermediates)

**Core Operations with Gradients**:
```rust
// All element-wise ops
add_backward, sub_backward, mul_backward, div_backward
exp_backward, log_backward, pow_backward, sqrt_backward
sin_backward, cos_backward, tanh_backward, sigmoid_backward

// Reductions
sum_backward, mean_backward, max_backward

// Linear algebra
matmul_backward, conv2d_backward

// Activations (all 14 GPU ops)
relu_backward, sigmoid_backward, gelu_backward, softmax_backward, etc.
```

**Memory Optimization**:
```rust
// Gradient checkpointing (save memory for deep networks)
let checkpoint = Checkpoint::new();
let y = checkpoint.run(|| {
    // Forward pass here (intermediates not saved)
    model.forward(x)
});

// In-place operations (where safe)
let mut x = Tensor::<f32, 2>::new([3, 3]);
x.relu_();  // In-place ReLU (saves memory, but cannot backward through this)
```

**Deliverables**:
- Reverse-mode AD engine
- Dynamic graph construction
- Gradient tape with backward functions
- All element-wise ops with gradients
- Matmul, conv2d with gradients
- All 14 activations with gradients
- Gradient checkpointing
- In-place operations

**Success Metrics**:
- Can train simple feedforward networks
- Gradients match PyTorch (< 1e-5 error)
- Memory-efficient gradient computation

#### v0.8.0: Neural Network Layers (3-4 months)

**nn::Module Trait**:
```rust
pub trait Module {
    type Input;
    type Output;

    fn forward(&self, input: Self::Input) -> Self::Output;
    fn parameters(&self) -> Vec<&Tensor>;
    fn zero_grad(&mut self);
}

// Example: Linear layer
pub struct Linear {
    weight: Tensor<f32, 2>,
    bias: Option<Tensor<f32, 1>>,
}

impl Module for Linear {
    type Input = Tensor<f32, 2>;
    type Output = Tensor<f32, 2>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        let output = input.matmul(&self.weight.transpose());
        if let Some(ref bias) = self.bias {
            output.add(bias)
        } else {
            output
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }
}
```

**Core Layers**:
```rust
// Fully connected
let linear = nn::Linear::new(784, 128);

// Convolutional
let conv2d = nn::Conv2d::new(3, 64, kernel_size=3, stride=1, padding=1);
let maxpool = nn::MaxPool2d::new(kernel_size=2, stride=2);

// Normalization
let batchnorm = nn::BatchNorm2d::new(64);
let layernorm = nn::LayerNorm::new(128);

// Regularization
let dropout = nn::Dropout::new(p=0.5);
```

**Sequential Container**:
```rust
let model = nn::Sequential::new()
    .add(nn::Linear::new(784, 256))
    .add(nn::ReLU::new())
    .add(nn::Dropout::new(0.5))
    .add(nn::Linear::new(256, 128))
    .add(nn::ReLU::new())
    .add(nn::Linear::new(128, 10));

let output = model.forward(input);
```

**Loss Functions**:
```rust
// Cross-entropy loss (classification)
let criterion = nn::CrossEntropyLoss::new();
let loss = criterion.forward(outputs, targets);

// MSE loss (regression)
let mse = nn::MSELoss::new();
let loss = mse.forward(predictions, targets);

// Binary cross-entropy (binary classification)
let bce = nn::BCELoss::new();
let loss = bce.forward(predictions, targets);

// Custom loss support
trait Loss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor;
}
```

**Deliverables**:
- nn::Module trait
- Linear (fully connected) layer
- Conv2d, MaxPool2d layers
- BatchNorm, LayerNorm
- Dropout
- Sequential container
- CrossEntropyLoss, MSELoss, BCELoss

**Success Metrics**:
- Can build ResNet-18, VGG-16
- Layer API matches PyTorch ergonomics
- Automatic parameter tracking

#### v0.9.0: Optimizers (2-3 months)

**Optimizer Trait**:
```rust
pub trait Optimizer {
    fn step(&mut self, params: &mut [Tensor]);
    fn zero_grad(&mut self, params: &mut [Tensor]);
}

// SGD with momentum
let optimizer = optim::SGD::new(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=true
);

// Training loop
for (inputs, targets) in dataloader {
    optimizer.zero_grad();
    let outputs = model.forward(inputs);
    let loss = criterion.forward(outputs, targets);
    loss.backward();
    optimizer.step();
}
```

**Core Optimizers**:
```rust
// SGD (with momentum, Nesterov)
let sgd = optim::SGD::new(params, lr=0.01, momentum=0.9, nesterov=true);

// Adam (with weight decay, AMSGrad)
let adam = optim::Adam::new(params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01);

// AdamW (decoupled weight decay)
let adamw = optim::AdamW::new(params, lr=0.001, weight_decay=0.01);

// RMSprop
let rmsprop = optim::RMSprop::new(params, lr=0.01, alpha=0.99);
```

**Learning Rate Schedulers**:
```rust
// Step decay
let scheduler = optim::StepLR::new(optimizer, step_size=30, gamma=0.1);

// Exponential decay
let scheduler = optim::ExponentialLR::new(optimizer, gamma=0.95);

// Cosine annealing
let scheduler = optim::CosineAnnealingLR::new(optimizer, T_max=100);

// Usage in training loop
for epoch in 0..num_epochs {
    train_one_epoch();
    scheduler.step();
}
```

**Deliverables**:
- Optimizer trait
- SGD (with momentum, Nesterov)
- Adam (with weight decay, AMSGrad)
- AdamW, RMSprop
- Learning rate schedulers (StepLR, ExponentialLR, CosineAnnealing)

**Success Metrics**:
- Can train ResNet-50 to convergence
- Optimizer behavior matches PyTorch
- Efficient gradient updates (in-place where possible)

#### v1.0.0: Training-Ready (3-4 months)

**Model Serialization**:
```rust
// Save checkpoint
let checkpoint = Checkpoint {
    model: model.state_dict(),
    optimizer: optimizer.state_dict(),
    epoch: 42,
    loss: 0.123,
};
checkpoint.save("model.pth")?;

// Load checkpoint
let checkpoint = Checkpoint::load("model.pth")?;
model.load_state_dict(checkpoint.model);
optimizer.load_state_dict(checkpoint.optimizer);

// ONNX export
model.export_onnx("model.onnx")?;
```

**Distributed Training** (Data Parallelism):
```rust
// Initialize distributed training
let world_size = 4;  // 4 GPUs
let rank = get_rank();

// Wrap model in DistributedDataParallel
let ddp_model = nn::parallel::DistributedDataParallel::new(model, rank);

// Training loop (gradients synchronized automatically)
for (inputs, targets) in dataloader {
    let outputs = ddp_model.forward(inputs);
    let loss = criterion.forward(outputs, targets);
    loss.backward();
    optimizer.step();  // Gradients averaged across GPUs
}
```

**Gradient Synchronization**:
- AllReduce for gradient averaging
- Efficient communication (NCCL-like)
- Fault tolerance (checkpoint/resume)

**Production Features**:
- Mixed precision training (FP16/BF16)
- Gradient accumulation
- Gradient clipping
- Early stopping
- TensorBoard logging

**Deliverables**:
- Model save/load (state_dict)
- ONNX export
- Distributed data parallelism
- Gradient synchronization
- Mixed precision support
- Production training utilities

**Performance Target**:
- 60-80% of PyTorch speed (within 20-40%)

**Success Metrics**:
- ✅ Train ResNet-50 on CIFAR-10 in <30 minutes (single GPU)
- ✅ 60-80% of PyTorch performance
- ✅ Used in production ML training pipelines

**Full MNIST/CIFAR-10 Examples**:
```rust
// Complete CIFAR-10 training example
use trueno::{nn, optim, data};

fn main() {
    // Load data
    let train_loader = data::DataLoader::new("cifar10", train=true, batch_size=128);
    let test_loader = data::DataLoader::new("cifar10", train=false, batch_size=128);

    // Build ResNet-18
    let model = nn::resnet18(num_classes=10);

    // Loss and optimizer
    let criterion = nn::CrossEntropyLoss::new();
    let optimizer = optim::Adam::new(model.parameters(), lr=0.001);

    // Training loop
    for epoch in 0..100 {
        for (images, labels) in train_loader {
            optimizer.zero_grad();
            let outputs = model.forward(images);
            let loss = criterion.forward(outputs, labels);
            loss.backward();
            optimizer.step();
        }

        // Validation
        let accuracy = evaluate(&model, &test_loader);
        println!("Epoch {}: Accuracy {:.2}%", epoch, accuracy * 100.0);
    }
}
```

---

## 4. Performance Targets

### 4.1 NumPy Performance Parity

**Target**: 80-120% of NumPy speed (within 20% either way)

**Benchmark Methodology**:
- Test sizes: 100, 1K, 10K, 100K, 1M, 10M elements
- Operations: add, mul, matmul, sum, mean, softmax
- Backends: Scalar, SIMD (AVX2), GPU
- Compare against: NumPy 1.26+ (with OpenBLAS)

**Expected Performance**:

| Operation | Size | NumPy (baseline) | Trueno CPU (SIMD) | Trueno GPU | Speedup |
|-----------|------|------------------|-------------------|------------|---------|
| add | 1K | 10 µs | 8 µs | - | 1.25x |
| add | 100K | 1 ms | 500 µs | - | 2x |
| add | 1M | 10 ms | 5 ms | 200 µs | 2x (CPU), 50x (GPU) |
| matmul | 128×128 | 200 µs | 180 µs | - | 1.1x |
| matmul | 1K×1K | 15 ms | 12 ms | 1.5 ms | 1.25x (CPU), 10x (GPU) |
| softmax | 10K | 120 µs | 100 µs | 60 µs | 1.2x (CPU), 2x (GPU) |
| softmax | 1M | 12 ms | 10 ms | 600 µs | 1.2x (CPU), 20x (GPU) |

**Key Insight**: GPU wins at >100K elements (transfer overhead amortized)

### 4.2 PyTorch Performance Parity

**Target**: 60-80% of PyTorch speed (within 20-40%)

**Benchmark Methodology**:
- Networks: ResNet-18, ResNet-50, VGG-16, Transformer (BERT-base)
- Tasks: CIFAR-10 training, ImageNet inference, NLP fine-tuning
- Hardware: NVIDIA A100 (GPU), AMD EPYC (CPU)
- Compare against: PyTorch 2.0+ (with cuDNN)

**Expected Performance**:

| Task | PyTorch (baseline) | Trueno | Speedup |
|------|-------------------|--------|---------|
| ResNet-50 training (CIFAR-10, 1 epoch) | 45s | 60s | 0.75x |
| ResNet-18 inference (ImageNet, batch=32) | 120 ms | 150 ms | 0.80x |
| BERT-base forward pass (seq_len=128) | 8 ms | 12 ms | 0.67x |
| Softmax (1M elements) | 800 µs | 600 µs | 1.33x |

**Key Insight**: Trueno may be faster for specific ops (softmax, activations) but slower for full models (cuDNN optimization)

---

## 5. API Compatibility Strategy

### 5.1 NumPy Compatibility

**Goal**: Minimize porting effort from NumPy → Trueno

**Approach 1: Idiomatic Rust API** (Primary)
```rust
// Rust-native API (ownership, error handling)
let a = Tensor::<f32, 2>::from_slice(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
let b = Tensor::<f32, 2>::from_slice(&[5.0, 6.0, 7.0, 8.0], [2, 2]);
let c = a.matmul(&b).unwrap();
```

**Approach 2: NumPy-like Macros** (Convenience)
```rust
// NumPy-style syntax via macros
let a = tensor![[1.0, 2.0], [3.0, 4.0]];
let b = tensor![[5.0, 6.0], [7.0, 8.0]];
let c = a.matmul(&b).unwrap();
```

**Approach 3: Function API** (Migration Path)
```rust
// NumPy function-style API
use trueno::np;

let a = np::array([[1.0, 2.0], [3.0, 4.0]]);
let b = np::array([[5.0, 6.0], [7.0, 8.0]]);
let c = np::matmul(&a, &b);
```

**Migration Guide**: Document NumPy → Trueno equivalents
```rust
// NumPy                          // Trueno
np.array([1, 2, 3])               Tensor::from_slice(&[1, 2, 3])
arr.reshape(2, 3)                 tensor.reshape([2, 3])
arr.transpose()                   tensor.transpose()
np.dot(a, b)                      a.dot(&b)
np.matmul(a, b)                   a.matmul(&b)
```

### 5.2 PyTorch Compatibility

**Goal**: Familiar API for PyTorch users

**Approach 1: Rust-native nn::Module** (Primary)
```rust
use trueno::nn;

struct MyModel {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl nn::Module for MyModel {
    type Input = Tensor<f32, 2>;
    type Output = Tensor<f32, 2>;

    fn forward(&self, x: Self::Input) -> Self::Output {
        let x = self.fc1.forward(x).relu();
        let x = self.fc2.forward(x);
        x.softmax(dim=1)
    }
}
```

**Approach 2: Macro-based Sequential** (Convenience)
```rust
let model = sequential![
    nn::Linear::new(784, 256),
    nn::ReLU::new(),
    nn::Linear::new(256, 10),
];
```

**Migration Guide**: Document PyTorch → Trueno equivalents
```rust
// PyTorch                        // Trueno
torch.tensor([1, 2, 3])           Tensor::from_slice(&[1, 2, 3])
x.requires_grad_(True)            x.requires_grad()
y.backward()                      y.backward()
x.grad                            x.grad()
nn.Linear(784, 128)               nn::Linear::new(784, 128)
F.relu(x)                         x.relu()
```

---

## 6. Use Case Validation

### 6.1 NumPy Replacement Use Cases

**Use Case 1: Scientific Computing** (Phase 2 - v0.6.0)
```rust
// Matrix operations for linear algebra
use trueno::Tensor;

let a = Tensor::<f64, 2>::random([1000, 1000]);
let b = Tensor::<f64, 2>::random([1000, 1000]);

// Solve Ax = b
let x = linalg::solve(&a, &b).unwrap();

// Eigenvalues/eigenvectors
let (eigenvalues, eigenvectors) = linalg::eig(&a).unwrap();
```

**Use Case 2: Data Preprocessing** (Phase 2 - v0.5.0)
```rust
// Normalize features for ML
let data = Tensor::<f32, 2>::from_csv("data.csv");
let mean = data.mean(axis=0, keepdims=true);
let std = data.std(axis=0, keepdims=true);
let normalized = (data - mean) / std;
```

**Use Case 3: Image Processing** (Phase 2 - v0.4.0)
```rust
// Apply filters to images
let image = Tensor::<f32, 3>::from_image("photo.jpg");  // (C, H, W)
let kernel = Tensor::<f32, 2>::gaussian_kernel(sigma=1.0);
let blurred = image.convolve2d(&kernel);
```

### 6.2 PyTorch Replacement Use Cases

**Use Case 1: Transfer Learning** (Phase 3 - v1.0.0)
```rust
// Fine-tune ResNet-50 on custom dataset
use trueno::{nn, optim, vision};

let mut model = vision::resnet50(pretrained=true);
model.fc = nn::Linear::new(2048, num_classes);  // Replace final layer

let criterion = nn::CrossEntropyLoss::new();
let optimizer = optim::Adam::new(model.parameters(), lr=0.001);

for epoch in 0..10 {
    for (images, labels) in train_loader {
        optimizer.zero_grad();
        let outputs = model.forward(images);
        let loss = criterion.forward(outputs, labels);
        loss.backward();
        optimizer.step();
    }
}
```

**Use Case 2: Custom Neural Network** (Phase 3 - v0.8.0)
```rust
// Build custom architecture
use trueno::nn;

struct Autoencoder {
    encoder: nn::Sequential,
    decoder: nn::Sequential,
}

impl nn::Module for Autoencoder {
    type Input = Tensor<f32, 2>;
    type Output = Tensor<f32, 2>;

    fn forward(&self, x: Self::Input) -> Self::Output {
        let encoded = self.encoder.forward(x);
        let decoded = self.decoder.forward(encoded);
        decoded
    }
}

let model = Autoencoder {
    encoder: sequential![
        nn::Linear::new(784, 256),
        nn::ReLU::new(),
        nn::Linear::new(256, 64),
    ],
    decoder: sequential![
        nn::Linear::new(64, 256),
        nn::ReLU::new(),
        nn::Linear::new(256, 784),
    ],
};
```

**Use Case 3: Inference Optimization** (Phase 3 - v0.8.0)
```rust
// Deploy trained model for fast inference
use trueno::nn;

let model = nn::load("model.pth").unwrap();
model.eval();  // Set to inference mode (disables dropout, etc.)

// Inference on GPU
let input = Tensor::<f32, 4>::from_image_batch(&images);  // (N, C, H, W)
let output = model.forward(input);  // GPU-accelerated
let predictions = output.argmax(dim=1);  // Get class predictions
```

---

## 7. Migration Path for Existing Projects

### 7.1 NumPy → Trueno Migration

**Step 1: Identify Vectorizable Code**
- Use trueno-analyze tool to scan Python codebase
- Identify NumPy operations that can be replaced

**Step 2: Incremental Replacement**
- Start with 1D operations (available today in v0.2.0)
- Use FFI to call Trueno from Python (via PyO3)
- Gradually replace more operations as Trueno gains features

**Step 3: Full Transpilation**
- Use Depyler (NumPy → Trueno transpiler) for automated conversion
- Manually verify correctness
- Benchmark performance improvements

**Example Migration**:
```python
# Before (NumPy)
import numpy as np

def process_signals(data):
    normalized = (data - data.mean()) / data.std()
    activated = 1.0 / (1.0 + np.exp(-normalized))
    return activated.sum()

# After (Trueno via PyO3)
import trueno_py

def process_signals(data):
    tensor = trueno_py.Tensor.from_numpy(data)
    normalized = tensor.normalize()
    activated = normalized.sigmoid()
    return activated.sum()
```

### 7.2 PyTorch → Trueno Migration

**Step 1: Identify Inference-Only Models**
- Target models that don't need retraining
- Export PyTorch model weights

**Step 2: Rewrite Forward Pass**
- Translate PyTorch forward() to Trueno (Phase 3)
- Load pretrained weights

**Step 3: Optimize Performance**
- Use Trueno GPU acceleration
- Benchmark against PyTorch inference

**Example Migration**:
```python
# Before (PyTorch)
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

# After (Trueno - Phase 3)
use trueno::nn;

struct SimpleNet {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl nn::Module for SimpleNet {
    type Input = Tensor<f32, 2>;
    type Output = Tensor<f32, 2>;

    fn forward(&self, x: Self::Input) -> Self::Output {
        let x = self.fc1.forward(x).relu();
        let x = self.fc2.forward(x);
        x.softmax(dim=1)
    }
}
```

---

## 8. Risk Analysis & Mitigation

### 8.1 Technical Risks

**Risk 1: GPU Performance Gap**
- **Threat**: Trueno GPU slower than PyTorch (cuDNN optimized)
- **Probability**: HIGH
- **Impact**: HIGH (undermines value proposition)
- **Mitigation**:
  - Focus on operations where we can win (activations, reductions)
  - Document where PyTorch is faster (cuDNN convolutions)
  - Target 60-80% of PyTorch speed (acceptable for Rust benefits)
  - Optimize critical kernels (matmul, conv2d)

**Risk 2: Autograd Complexity**
- **Threat**: Reverse-mode AD too complex to implement correctly
- **Probability**: MEDIUM
- **Impact**: HIGH (blocks training entirely)
- **Mitigation**:
  - Study PyTorch autograd internals
  - Start with simple ops (add, mul) before complex (matmul, conv)
  - Comprehensive gradient checking tests
  - Hire autograd expert if needed

**Risk 3: Memory Overhead**
- **Threat**: Autograd graph consumes excessive memory
- **Probability**: MEDIUM
- **Impact**: MEDIUM (limits model size)
- **Mitigation**:
  - Implement gradient checkpointing
  - Release intermediates eagerly
  - Provide in-place operations where safe
  - Profile memory usage, optimize hotspots

### 8.2 Strategic Risks

**Risk 1: PyTorch Ecosystem Lock-in**
- **Threat**: Users prefer PyTorch due to ecosystem (libraries, pretrained models)
- **Probability**: HIGH
- **Impact**: MEDIUM (limits adoption)
- **Mitigation**:
  - Provide ONNX import/export
  - Load PyTorch pretrained weights
  - Target Rust-native ML use cases (embedded, production)
  - Emphasize benefits: type safety, performance, no Python overhead

**Risk 2: Feature Creep**
- **Threat**: Trying to match every PyTorch feature delays v1.0
- **Probability**: MEDIUM
- **Impact**: HIGH (never reach production-ready state)
- **Mitigation**:
  - Focus on 80% use cases (ResNet, BERT, simple CNNs)
  - Explicitly declare non-goals (symbolic computation, JIT compilation)
  - Ship v1.0 with core features, add advanced features in v1.x

**Risk 3: Rust Ergonomics**
- **Threat**: Rust's ownership model makes ML code verbose/difficult
- **Probability**: MEDIUM
- **Impact**: MEDIUM (poor developer experience)
- **Mitigation**:
  - Provide ergonomic macros (tensor!, sequential!)
  - Smart default behaviors (auto-grad on by default)
  - Comprehensive examples and tutorials
  - Gather user feedback, iterate on API

---

## 9. Success Metrics & KPIs

### 9.1 Technical Metrics

**Phase 1 (v0.3.0) - 1D Operations Complete**:
- ✅ All common 1D operations GPU-accelerated
- ✅ 10-50x GPU speedup validated by benchmarks
- ✅ >90% test coverage
- ✅ Mutation testing ≥80% kill rate

**Phase 2 (v0.6.0) - NumPy Parity**:
- ✅ 80-120% of NumPy performance
- ✅ Pass 80%+ of NumPy test suite (for covered ops)
- ✅ Multi-dimensional arrays (0D-4D)
- ✅ Broadcasting support

**Phase 3 (v1.0.0) - Training Ready**:
- ✅ Train ResNet-50 on CIFAR-10 in <30 minutes (single GPU)
- ✅ 60-80% of PyTorch performance
- ✅ Autograd matches PyTorch (< 1e-5 gradient error)

### 9.2 Adoption Metrics

**Phase 1**:
- ✅ Used in production by ≥3 projects
- ✅ ≥100 GitHub stars
- ✅ ≥10 contributors

**Phase 2**:
- ✅ ≥10 production deployments
- ✅ ≥500 GitHub stars
- ✅ ≥50 contributors

**Phase 3**:
- ✅ Used in production ML training pipelines
- ✅ ≥1,000 GitHub stars
- ✅ ≥100 contributors
- ✅ Featured in Rust ML blog posts/talks

### 9.3 Quality Metrics (Continuous)

**EXTREME TDD Standards** (All phases):
- Test coverage: ≥90%
- Mutation testing: ≥80% kill rate
- Property-based tests: All core operations
- Backend equivalence: GPU vs SIMD vs Scalar (< 1e-5 error)
- Zero clippy warnings
- PMAT TDG: ≥B+ (85/100)

---

## 10. Non-Goals

### What Trueno Will NOT Be

❌ **100% PyTorch-compatible** - Inspired by, not clone of
- We will NOT replicate every PyTorch feature
- Focus on core 80% use cases (ResNet, BERT, simple CNNs)
- Explicitly omit: JIT compilation, symbolic computation, TorchScript

❌ **Research-first library** - Production performance is priority
- We will NOT prioritize cutting-edge research features
- Focus on battle-tested architectures
- Add experimental features only after proven in production

❌ **Python-first** - Rust-native, Python bindings optional
- Primary API is Rust, not Python
- PyO3 bindings for interop, but Rust is first-class
- Performance over Python compatibility

❌ **Dynamic typing** - Static typing for safety
- Compile-time shape checking where possible
- Type errors at compile time, not runtime
- Trade ergonomics for safety/performance

❌ **Symbolic computation** - Eager execution only
- No graph optimization, no lazy evaluation
- Simple mental model: operations execute immediately
- Focus on imperative PyTorch-style API

---

## 11. Appendix: Competitive Analysis

### 11.1 Trueno vs. NumPy

| Feature | NumPy | Trueno v0.2.0 | Trueno v0.6.0 (Target) |
|---------|-------|---------------|------------------------|
| **Multi-dim arrays** | ✅ 0D-32D | ❌ 1D only | ✅ 0D-4D |
| **Broadcasting** | ✅ Full | ❌ None | ✅ Full |
| **Data types** | ✅ 20+ | ❌ f32 only | ✅ 10+ (f16/f32/f64/i32/etc.) |
| **GPU acceleration** | ❌ CPU only | ✅ 14 ops | ✅ 50+ ops |
| **Performance (SIMD)** | ❌ Limited | ✅ 2-8x faster | ✅ 2-8x faster |
| **Performance (GPU)** | ❌ N/A | ✅ 10-50x faster | ✅ 10-50x faster |
| **Type safety** | ❌ Runtime | ✅ Compile-time | ✅ Compile-time |
| **Memory safety** | ❌ C backend | ✅ Rust safe | ✅ Rust safe |

**Verdict**: Trueno v0.6.0 will match NumPy functionality with superior performance and safety.

### 11.2 Trueno vs. PyTorch

| Feature | PyTorch | Trueno v0.2.0 | Trueno v1.0.0 (Target) |
|---------|---------|---------------|------------------------|
| **Autograd** | ✅ Full | ❌ None | ✅ Full |
| **GPU ops** | ✅ 1000+ | ✅ 14 | ✅ 100+ |
| **Training** | ✅ Full | ❌ Inference only | ✅ Full |
| **Layers** | ✅ 100+ | ❌ None | ✅ 20+ core layers |
| **Optimizers** | ✅ 10+ | ❌ None | ✅ 5+ (SGD, Adam, AdamW, etc.) |
| **Pretrained models** | ✅ 1000+ | ❌ None | ✅ 10+ (ResNet, BERT, etc.) |
| **Performance (cuDNN)** | ✅ Baseline | ❌ N/A | ✅ 60-80% |
| **Type safety** | ❌ Runtime | ✅ Compile-time | ✅ Compile-time |
| **Memory safety** | ❌ Python/C++ | ✅ Rust safe | ✅ Rust safe |

**Verdict**: Trueno v1.0.0 will enable training with 60-80% of PyTorch performance, prioritizing safety.

### 11.3 Trueno vs. Other Rust ML Libraries

**vs. burn** (Rust deep learning):
- burn: Dynamic graphs, flexible, but less mature
- Trueno: Focus on production performance, GPU optimization, NumPy/PyTorch API

**vs. candle** (HuggingFace):
- candle: Inference-focused, minimal dependencies
- Trueno: Training + inference, SIMD + GPU, broader API surface

**vs. ndarray** (Rust NumPy):
- ndarray: CPU-only, no GPU, no autograd
- Trueno: GPU-accelerated, autograd (Phase 3), training-ready

**Trueno Differentiation**:
- ✅ GPU-first design (14 ops today, 100+ by v1.0)
- ✅ EXTREME TDD (>90% coverage, mutation testing)
- ✅ PyTorch/NumPy compatibility (familiar API)
- ✅ Production focus (safety, performance, PMAT quality gates)

---

## 12. Conclusion

Trueno is strategically positioned to become the **de facto PyTorch/NumPy replacement for Rust**. This specification defines a clear, achievable roadmap:

**v0.2.0 → v0.3.0** (2-3 months): Complete 1D operations, validate GPU performance
**v0.3.0 → v0.6.0** (6-12 months): Multi-dimensional tensors, broadcasting, NumPy parity
**v0.6.0 → v1.0.0** (12-18 months): Autograd, training, PyTorch parity

**Total timeline**: 20-33 months to production-ready deep learning library.

**Key Success Factors**:
1. **Focus**: Prioritize 80% use cases, defer advanced features
2. **Quality**: Maintain EXTREME TDD standards throughout
3. **Performance**: Validate GPU speedup claims, optimize critical paths
4. **Adoption**: Ship early, gather feedback, iterate on API

**Next Steps**:
1. ✅ Complete v0.2.1 (softmax/log_softmax GPU) - DONE
2. Benchmark all GPU ops (validate 10-50x claims)
3. Implement remaining activations (hardswish, mish, selu)
4. Begin Phase 2 design (Tensor<T, N> type)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-17
**Status**: Living Document (update as roadmap evolves)
**Owner**: Trueno Core Team
