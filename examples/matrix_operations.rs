//! Matrix Operations Example for Trueno
//!
//! Demonstrates the matrix operations available in Trueno including:
//! - Matrix construction and basic operations
//! - Matrix multiplication (matmul)
//! - Matrix transpose
//! - Matrix-vector operations (matvec, vecmat)
//!
//! Run with: cargo run --example matrix_operations

use trueno::{Matrix, Vector};

fn main() {
    println!("🧮 Trueno Matrix Operations Demo");
    println!("=================================\n");

    // ========================================================================
    // Matrix Construction
    // ========================================================================
    println!("📐 Matrix Construction");
    println!("----------------------\n");

    let m1 = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("Example should not fail");
    println!("Matrix m1 (2×3):");
    print_matrix(&m1);

    let m2 = Matrix::identity(3);
    println!("Identity matrix I₃ (3×3):");
    print_matrix(&m2);

    let m3 = Matrix::zeros(3, 2);
    println!("Zero matrix 0₃ₓ₂ (3×2):");
    print_matrix(&m3);

    // ========================================================================
    // Matrix Multiplication
    // ========================================================================
    println!("\n📊 Matrix Multiplication (matmul)");
    println!("----------------------------------\n");

    let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("Example should not fail");
    let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        .expect("Example should not fail");

    println!("Matrix A (2×3):");
    print_matrix(&a);
    println!("Matrix B (3×2):");
    print_matrix(&b);

    let c = a.matmul(&b).expect("Example should not fail");
    println!("A × B (2×2):");
    print_matrix(&c);
    println!("Calculation:");
    println!("  C[0,0] = 1×7 + 2×9 + 3×11 = 58");
    println!("  C[0,1] = 1×8 + 2×10 + 3×12 = 64");
    println!("  C[1,0] = 4×7 + 5×9 + 6×11 = 139");
    println!("  C[1,1] = 4×8 + 5×10 + 6×12 = 154");

    // ========================================================================
    // Matrix Transpose
    // ========================================================================
    println!("\n🔄 Matrix Transpose");
    println!("-------------------\n");

    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("Example should not fail");
    println!("Original matrix M (2×3):");
    print_matrix(&m);

    let m_t = m.transpose();
    println!("Transposed M^T (3×2):");
    print_matrix(&m_t);
    println!("Properties:");
    println!("  • Rows and columns swapped: 2×3 → 3×2");
    println!("  • Element M[i,j] becomes M^T[j,i]");
    println!("  • (M^T)^T = M");

    // ========================================================================
    // Matrix-Vector Multiplication
    // ========================================================================
    println!("\n🎯 Matrix-Vector Multiplication (matvec)");
    println!("-----------------------------------------\n");

    let matrix = Matrix::from_vec(
        3,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .expect("Example should not fail");
    let vector = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);

    println!("Matrix A (3×4):");
    print_matrix(&matrix);
    println!("Vector v (4×1):");
    print_vector(&vector);

    let result = matrix.matvec(&vector).expect("Example should not fail");
    println!("A × v (3×1):");
    print_vector(&result);
    println!("Calculation:");
    println!("  result[0] = 1×1 + 2×2 + 3×3 + 4×4 = 30");
    println!("  result[1] = 5×1 + 6×2 + 7×3 + 8×4 = 70");
    println!("  result[2] = 9×1 + 10×2 + 11×3 + 12×4 = 110");

    // ========================================================================
    // Vector-Matrix Multiplication
    // ========================================================================
    println!("\n🎯 Vector-Matrix Multiplication (vecmat)");
    println!("-----------------------------------------\n");

    let matrix2 = Matrix::from_vec(
        3,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .expect("Example should not fail");
    let vector2 = Vector::from_slice(&[1.0, 2.0, 3.0]);

    println!("Vector v^T (1×3):");
    print_vector(&vector2);
    println!("Matrix A (3×4):");
    print_matrix(&matrix2);

    let result2 = Matrix::vecmat(&vector2, &matrix2).expect("Example should not fail");
    println!("v^T × A (1×4):");
    print_vector(&result2);
    println!("Calculation:");
    println!("  result[0] = 1×1 + 2×5 + 3×9 = 38");
    println!("  result[1] = 1×2 + 2×6 + 3×10 = 44");
    println!("  result[2] = 1×3 + 2×7 + 3×11 = 50");
    println!("  result[3] = 1×4 + 2×8 + 3×12 = 56");

    // ========================================================================
    // Neural Network Linear Layer Example
    // ========================================================================
    println!("\n🧠 Real-World Use Case: Neural Network Linear Layer");
    println!("----------------------------------------------------\n");

    // Simulate a simple linear layer: y = W×x + b
    // where W is a 3×4 weight matrix, x is a 4D input, b is a 3D bias
    let weights = Matrix::from_vec(
        3,
        4,
        vec![
            0.1, 0.2, -0.1, 0.3, // neuron 1 weights
            -0.2, 0.1, 0.4, -0.1, // neuron 2 weights
            0.3, -0.1, 0.2, 0.1, // neuron 3 weights
        ],
    )
    .expect("Example should not fail");
    let input = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let bias = Vector::from_slice(&[0.1, -0.1, 0.2]);

    println!("Weight matrix W (3×4):");
    print_matrix(&weights);
    println!("Input vector x (4D):");
    print_vector(&input);
    println!("Bias vector b (3D):");
    print_vector(&bias);

    let wx = weights.matvec(&input).expect("Example should not fail");
    let output = wx.add(&bias).expect("Example should not fail");

    println!("Linear layer output y = W×x + b:");
    print_vector(&output);
    println!("  → This becomes the input to the activation function");
    println!("  → Common activations: ReLU, sigmoid, tanh, softmax");

    // ========================================================================
    // Batch Processing with vecmat
    // ========================================================================
    println!("\n📦 Batch Processing: Multiple Inputs");
    println!("-------------------------------------\n");

    println!("Processing 3 samples through the same linear layer:");
    let samples = [
        Vector::from_slice(&[1.0, 0.0, 0.0, 0.0]),
        Vector::from_slice(&[0.0, 1.0, 0.0, 0.0]),
        Vector::from_slice(&[0.0, 0.0, 1.0, 0.0]),
    ];

    for (i, sample) in samples.iter().enumerate() {
        let wx = weights.matvec(sample).expect("Example should not fail");
        let output = wx.add(&bias).expect("Example should not fail");
        println!("  Sample {}: {:?}", i + 1, output.as_slice());
    }

    // ========================================================================
    // Batched Matrix Multiplication (3D Tensors)
    // ========================================================================
    println!("\n🔢 Batched Matrix Multiplication (3D Tensors)");
    println!("-----------------------------------------------\n");

    // Simulate batch of 2 matrix multiplications: [2, 3, 4] @ [2, 4, 2] -> [2, 3, 2]
    let batch = 2;
    let m = 3;
    let k = 4;
    let n = 2;

    // Batch 0: 3×4 matrix, Batch 1: 3×4 matrix
    let a_data: Vec<f32> = (0..batch * m * k).map(|i| i as f32 * 0.1).collect();
    // Batch 0: 4×2 matrix, Batch 1: 4×2 matrix
    let b_data: Vec<f32> = (0..batch * k * n).map(|i| (i as f32 + 1.0) * 0.1).collect();

    println!("Shape: [batch={}, m={}, k={}] @ [batch={}, k={}, n={}]", batch, m, k, batch, k, n);
    println!("A data (flattened): {:?}", &a_data[..8]);
    println!("B data (flattened): {:?}", &b_data[..8]);

    let result = Matrix::batched_matmul(&a_data, &b_data, batch, m, k, n)
        .expect("Batched matmul should succeed");

    println!("Output shape: [batch={}, m={}, n={}]", batch, m, n);
    println!("Result (flattened): {:?}", &result[..6]);
    println!("  → Each batch processed independently using SIMD matmul");

    // ========================================================================
    // Batched Matrix Multiplication (4D Tensors - Attention Pattern)
    // ========================================================================
    println!("\n🧠 Batched 4D Matrix Multiplication (Attention Pattern)");
    println!("--------------------------------------------------------\n");

    // Simulate multi-head attention: Q @ K^T
    // Shape: [batch=1, heads=2, seq=4, head_dim=8] @ [batch=1, heads=2, head_dim=8, seq=4]
    let batch = 1;
    let heads = 2;
    let seq_len = 4;
    let head_dim = 8;

    // Q: [1, 2, 4, 8] - 64 elements
    let q_data: Vec<f32> = (0..batch * heads * seq_len * head_dim)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    // K^T: [1, 2, 8, 4] - 64 elements
    let kt_data: Vec<f32> = (0..batch * heads * head_dim * seq_len)
        .map(|i| (i as f32 * 0.02).cos())
        .collect();

    println!("Multi-head attention pattern: Q @ K^T");
    println!("  Q shape: [batch={}, heads={}, seq={}, head_dim={}]", batch, heads, seq_len, head_dim);
    println!("  K^T shape: [batch={}, heads={}, head_dim={}, seq={}]", batch, heads, head_dim, seq_len);

    let attn_scores = Matrix::batched_matmul_4d(
        &q_data,
        &kt_data,
        batch,
        heads,
        seq_len,   // m
        head_dim,  // k
        seq_len,   // n
    )
    .expect("4D batched matmul should succeed");

    println!("  Output shape: [batch={}, heads={}, seq={}, seq={}]", batch, heads, seq_len, seq_len);
    println!("  Attention scores (first 8): {:?}", &attn_scores[..8]);
    println!("  → Used for transformer attention: softmax(Q @ K^T / sqrt(d)) @ V");

    // ========================================================================
    // Mathematical Properties
    // ========================================================================
    println!("\n✅ Verified Mathematical Properties");
    println!("------------------------------------\n");

    let test_m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("Example should not fail");
    let test_v = Vector::from_slice(&[5.0, 6.0]);

    // Identity property: I×v = v
    let identity = Matrix::identity(2);
    let iv = identity.matvec(&test_v).expect("Example should not fail");
    assert_eq!(iv.as_slice(), test_v.as_slice());
    println!("✓ Identity: I×v = v");

    // Transpose property: (A×v)^T has same values as v^T×A^T
    let av = test_m.matvec(&test_v).expect("Example should not fail");
    let m_t = test_m.transpose();
    let v_mt = Matrix::vecmat(&test_v, &m_t).expect("Example should not fail");
    assert_eq!(av.as_slice(), v_mt.as_slice());
    println!("✓ Transpose: (A×v)^T = v^T×A^T");

    // Zero property: A×0 = 0
    let zero_v = Vector::from_slice(&[0.0, 0.0]);
    let result = test_m.matvec(&zero_v).expect("Example should not fail");
    assert_eq!(result.as_slice(), &[0.0, 0.0]);
    println!("✓ Zero: A×0 = 0");

    println!("\n🎉 All matrix operations working correctly!");
    println!("\n📚 For more examples, see:");
    println!("   • examples/activation_functions.rs - Neural network activations");
    println!("   • examples/ml_similarity.rs - ML vector operations");
    println!("   • examples/performance_demo.rs - SIMD performance");
}

/// Helper function to print a matrix in a readable format
fn print_matrix(m: &Matrix<f32>) {
    let (rows, cols) = m.shape();
    for i in 0..rows {
        print!("  [");
        for j in 0..cols {
            if j > 0 {
                print!(", ");
            }
            print!("{:6.1}", m.get(i, j).expect("Example should not fail"));
        }
        println!("]");
    }
}

/// Helper function to print a vector in a readable format
fn print_vector(v: &Vector<f32>) {
    print!("  [");
    for (i, val) in v.as_slice().iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:6.1}", val);
    }
    println!("]");
}
