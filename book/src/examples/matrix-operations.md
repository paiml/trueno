# Matrix Operations

This chapter demonstrates Trueno's matrix operations using the `matrix_operations` example.

## Running the Example

```bash
cargo run --example matrix_operations
```

## Basic Matrix Operations

### Creating Matrices

```rust
use trueno::Matrix;

// Create from row-major data
let a = Matrix::from_vec(2, 3, vec![
    1.0, 2.0, 3.0,  // Row 0
    4.0, 5.0, 6.0,  // Row 1
])?;

// Identity matrix
let identity = Matrix::identity(3);

// Zero matrix
let zeros = Matrix::zeros(2, 3);
```

### Matrix Multiplication

```rust
// C = A × B
let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;

let c = a.matmul(&b)?;  // Result: 2×2 matrix
```

### Matrix-Vector Multiplication

```rust
use trueno::{Matrix, Vector};

let weights = Matrix::from_vec(3, 4, weight_data)?;
let input = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);

let output = weights.matvec(&input)?;  // Result: Vector of length 3
```

### Transpose

```rust
let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
let at = a.transpose();  // Result: 3×2 matrix
```

## Neural Network Layers

### Linear Layer (Dense)

```rust
fn linear_layer(
    input: &Vector,
    weights: &Matrix,
    bias: &Vector,
) -> Result<Vector, TruenoError> {
    let output = weights.matvec(input)?;
    output.add(bias)
}
```

### Batch Processing

```rust
// Process multiple samples through the same layer
let samples = vec![
    Vector::from_slice(&[0.2, -0.3, 0.5]),
    Vector::from_slice(&[0.3, 0.0, 0.1]),
    Vector::from_slice(&[0.0, 0.3, 0.4]),
];

for sample in &samples {
    let output = weights.matvec(sample)?;
    println!("Output: {:?}", output.as_slice());
}
```

## Mathematical Properties

The example verifies key mathematical properties:

### Identity Property

```rust
// I × v = v
let identity = Matrix::identity(3);
let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
let result = identity.matvec(&v)?;
assert_eq!(result.as_slice(), v.as_slice());
```

### Transpose Property

```rust
// (A × v)^T = v^T × A^T
// This is verified in the example
```

### Zero Property

```rust
// A × 0 = 0
let zeros = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0]);
let result = weights.matvec(&zeros)?;
// All elements should be 0
```

## Performance Considerations

### Blocking for Cache Efficiency

Trueno uses blocked matrix multiplication for better cache utilization:

```rust
// Automatic blocking for large matrices
let large_a = Matrix::from_vec(1024, 1024, data_a)?;
let large_b = Matrix::from_vec(1024, 1024, data_b)?;
let c = large_a.matmul(&large_b)?;  // Uses tiled algorithm internally
```

### SIMD Acceleration

Matrix operations automatically use SIMD when beneficial:

- AVX2: Process 8 f32 values per instruction
- AVX-512: Process 16 f32 values per instruction
- Automatic fallback to scalar for small matrices

### GPU Acceleration

For large matrices, enable GPU acceleration:

```bash
cargo run --release --features gpu --example matrix_operations
```

The GPU backend is automatically selected for matrices above the threshold (typically 256×256).

## Benchmark Suite

Run the matrix benchmark suite:

```bash
cargo run --release --example benchmark_matrix_suite
```

This compares:
- Naive O(n³) multiplication
- SIMD-optimized blocked multiplication
- Parallel (rayon) multiplication

## See Also

- [Eigendecomposition](../api-reference/eigendecomposition.md) - Symmetric eigenvalue decomposition
- [GPU Backend](../architecture/gpu-backend.md) - GPU acceleration details
- [Performance Targets](../performance/targets.md) - Expected speedups
