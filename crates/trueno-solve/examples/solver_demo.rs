//! Solver demonstration: LU, QR, SVD, Cholesky.
//!
//! ```sh
//! cargo run --example solver_demo -p trueno-solve
//! ```

use trueno_solve::{cholesky, lu_factorize, qr_factorize, svd};

fn main() {
    println!("=== trueno-solve: Dense Solver Demo ===\n");

    // 1. LU factorization
    println!("--- LU Factorization ---");
    let a = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0_f32];
    let b = [4.0, 7.0, 10.0_f32];
    let lu = lu_factorize(&a, 3).expect("LU ok");
    let x = lu.solve(&b).expect("solve ok");
    println!("A = [[2,1,0],[1,3,1],[0,1,4]]");
    println!("b = [4, 7, 10]");
    println!("x = [{:.4}, {:.4}, {:.4}]", x[0], x[1], x[2]);

    // Verify
    let mut residual = 0.0f32;
    for i in 0..3 {
        let mut ax = 0.0f32;
        for j in 0..3 {
            ax += a[i * 3 + j] * x[j];
        }
        residual += (ax - b[i]).powi(2);
    }
    println!("||Ax - b|| = {:.2e}\n", residual.sqrt());

    // 2. QR factorization
    println!("--- QR Factorization ---");
    let a_rect = [1.0, 1.0, 1.0, 2.0, 1.0, 3.0_f32]; // 3×2
    let qr = qr_factorize(&a_rect, 3, 2).expect("QR ok");
    let x_ls = qr.solve(&[1.0, 2.0, 3.0]).expect("solve ok");
    println!("Least-squares: x = [{:.4}, {:.4}]", x_ls[0], x_ls[1]);

    // 3. SVD
    println!("\n--- SVD ---");
    let a_svd = [3.0, 2.0, 2.0, 3.0_f32];
    let result = svd(&a_svd, 2, 2).expect("SVD ok");
    println!("A = [[3,2],[2,3]]");
    println!(
        "Singular values: [{:.4}, {:.4}]",
        result.sigma[0], result.sigma[1]
    );
    println!("Expected: [5.0, 1.0]");

    // 4. Cholesky
    println!("\n--- Cholesky ---");
    let a_spd = [4.0, 2.0, 2.0, 3.0_f32];
    let chol = cholesky(&a_spd, 2).expect("Cholesky ok");
    let x_chol = chol.solve(&[8.0, 7.0]).expect("solve ok");
    println!("A = [[4,2],[2,3]], b = [8,7]");
    println!("x = [{:.4}, {:.4}]", x_chol[0], x_chol[1]);

    println!("\n=== All solver demos passed ===");
}
