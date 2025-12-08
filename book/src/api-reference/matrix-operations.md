# Matrix Operations

The `Matrix<T>` type provides 2D matrix operations with SIMD acceleration.

## Creating Matrices

```rust
use trueno::Matrix;

// From dimensions (uninitialized)
let m = Matrix::<f32>::new(3, 4);

// From Vec with dimensions
let m = Matrix::<f32>::from_vec(2, 3, vec![
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
])?;

// Special matrices
let zeros = Matrix::<f32>::zeros(3, 3);
let identity = Matrix::<f32>::identity(4);
```

## Basic Properties

```rust
let m = Matrix::<f32>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;

m.rows();        // 2
m.cols();        // 3
m.len();         // 6 (total elements)
m.as_slice();    // &[f32] view of data
m.get(0, 1);     // Some(2.0)
m.get_mut(1, 2); // Mutable access
```

## Matrix Multiplication

```rust
let a = Matrix::<f32>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
let b = Matrix::<f32>::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;

// Matrix-matrix multiplication: [2×3] × [3×2] = [2×2]
let c = a.matmul(&b)?;
```

## Matrix-Vector Multiplication

```rust
use trueno::Vector;

let m = Matrix::<f32>::from_vec(3, 4, vec![/* 12 elements */])?;
let v = Vector::<f32>::from_slice(&[1.0, 2.0, 3.0, 4.0]);

// Matrix × Vector: [3×4] × [4×1] = [3×1]
let result = m.matvec(&v)?;

// Vector × Matrix: [1×3] × [3×4] = [1×4]
let v2 = Vector::<f32>::from_slice(&[1.0, 2.0, 3.0]);
let result = m.vecmat(&v2)?;
```

## Transpose

```rust
let m = Matrix::<f32>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;

// [2×3] → [3×2]
let mt = m.transpose();
```

## Convolution (2D)

For image processing and CNNs.

```rust
let image = Matrix::<f32>::from_vec(5, 5, /* 25 elements */)?;
let kernel = Matrix::<f32>::from_vec(3, 3, vec![
    1.0, 0.0, -1.0,
    2.0, 0.0, -2.0,
    1.0, 0.0, -1.0,
])?; // Sobel edge detection

let edges = image.convolve2d(&kernel)?;
```

## Embedding Lookup

For NLP models (word embeddings).

```rust
// Embedding table: vocab_size × embedding_dim
let embeddings = Matrix::<f32>::from_vec(1000, 128, /* ... */)?;

// Token indices
let tokens: Vec<usize> = vec![42, 7, 256, 13];

// Lookup: returns [4×128] matrix
let token_embeddings = embeddings.embedding_lookup(&tokens)?;
```

## GPU Acceleration

For large matrices, use the GPU backend.

```rust
use trueno::GpuBackend;

let mut gpu = GpuBackend::new();
let a = Matrix::<f32>::from_vec(1024, 1024, /* ... */)?;
let b = Matrix::<f32>::from_vec(1024, 1024, /* ... */)?;

// GPU-accelerated matmul
let c = gpu.matmul(&a, &b)?;
```

## Performance Tips

1. **Matrix multiplication**: O(n³) - GPU beneficial for n > 500
2. **Convolution**: Use separable kernels when possible
3. **Memory layout**: Row-major storage for cache efficiency
4. **Batch operations**: Group small matrices for GPU efficiency

See the [GPU Performance Guide](../performance/gpu-performance.md) for details.
