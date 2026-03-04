//! Sparse SpMV example — CUDA-parity-spec Phase 1
//!
//! Demonstrates CSR matrix construction, COO→CSR conversion, and SpMV.
//!
//! ```bash
//! cargo run -p trueno-sparse --example sparse_spmv
//! ```

use trueno_sparse::{CooMatrix, CsrMatrix, SparseOps};

fn main() {
    println!("=== Trueno Sparse: SpMV Example ===\n");

    // Build a sparse matrix from COO triplets
    // A = [[1, 0, 2, 0],
    //      [0, 3, 0, 0],
    //      [4, 0, 5, 6],
    //      [0, 0, 0, 7]]
    let coo = CooMatrix::new(
        4,
        4,
        vec![0, 0, 1, 2, 2, 2, 3],
        vec![0, 2, 1, 0, 2, 3, 3],
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    )
    .expect("Valid COO");

    println!("COO matrix: {} rows x {} cols, {} nonzeros", coo.rows, coo.cols, coo.nnz());

    // Convert to CSR
    let csr = CsrMatrix::from_coo(&coo);
    println!(
        "CSR matrix: {} rows x {} cols, {} nonzeros",
        csr.rows(),
        csr.cols(),
        csr.nnz()
    );
    println!("  avg nnz/row: {:.2}", csr.avg_nnz_per_row());
    println!("  row length variance: {:.4}", csr.row_length_variance());
    println!("  offsets: {:?}", csr.offsets());
    println!("  col_indices: {:?}", csr.col_indices());
    println!("  values: {:?}\n", csr.values());

    // SpMV: y = A * x
    let x = vec![1.0_f32, 2.0, 3.0, 4.0];
    let mut y = vec![0.0_f32; 4];

    csr.spmv(1.0, &x, 0.0, &mut y)
        .expect("SpMV dimension check");

    println!("SpMV: y = A * x");
    println!("  x = {:?}", x);
    println!("  y = {:?}", y);
    println!("  expected: [7.0, 6.0, 43.0, 28.0]");

    // Verify
    let expected = [7.0_f32, 6.0, 43.0, 28.0];
    for i in 0..4 {
        assert!(
            (y[i] - expected[i]).abs() < 1e-5,
            "Mismatch at y[{i}]: got {}, expected {}",
            y[i],
            expected[i]
        );
    }
    println!("\n  All values match. SpMV correct.");

    // SpMV with alpha/beta: y = 2*A*x + 0.5*y
    let mut y2 = vec![10.0_f32; 4];
    csr.spmv(2.0, &x, 0.5, &mut y2)
        .expect("SpMV alpha/beta");

    println!("\nSpMV: y = 2*A*x + 0.5*[10,10,10,10]");
    println!("  y = {:?}", y2);

    // Identity matrix test
    let eye = CsrMatrix::<f32>::identity(4);
    let mut y3 = vec![0.0_f32; 4];
    eye.spmv(1.0, &x, 0.0, &mut y3).expect("Identity SpMV");
    println!("\nIdentity SpMV: y = I * x = {:?}", y3);
    assert_eq!(y3, x);
    println!("  Identity check passed.");

    println!("\n=== Done ===");
}
