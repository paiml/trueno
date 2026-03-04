//! Solver demonstration: LU, QR, SVD, Cholesky, TRSM, BLAS Level-3.
//!
//! ```sh
//! cargo run --example solver_demo -p trueno-solve
//! ```

use trueno_solve::{cholesky, lu_factorize, qr_factorize, svd, syr2k, syrk, symm, trmm, trsm};
use trueno_solve::{DiagonalType, TriangularSide};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== trueno-solve: Full Solver Demo ===\n");

    // ── LU ─────────────────────────────────────────────────
    println!("--- LU Factorization ---");
    let a = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0_f32];
    let lu = lu_factorize(&a, 3)?;
    let x = lu.solve(&[4.0, 7.0, 10.0])?;
    println!("Ax=b: x = [{:.4}, {:.4}, {:.4}]", x[0], x[1], x[2]);

    // ── QR ─────────────────────────────────────────────────
    println!("\n--- QR Factorization ---");
    let qr = qr_factorize(&[1.0, 1.0, 1.0, 2.0, 1.0, 3.0_f32], 3, 2)?;
    let x_ls = qr.solve(&[1.0, 2.0, 3.0])?;
    println!("Least-squares: x = [{:.4}, {:.4}]", x_ls[0], x_ls[1]);

    // ── SVD ────────────────────────────────────────────────
    println!("\n--- SVD ---");
    let result = svd(&[3.0, 2.0, 2.0, 3.0_f32], 2, 2)?;
    println!("Singular values: [{:.4}, {:.4}]", result.sigma[0], result.sigma[1]);

    // ── Cholesky ───────────────────────────────────────────
    println!("\n--- Cholesky ---");
    let chol = cholesky(&[4.0, 2.0, 2.0, 3.0_f32], 2)?;
    let x_chol = chol.solve(&[8.0, 7.0])?;
    println!("x = [{:.4}, {:.4}]", x_chol[0], x_chol[1]);

    // ── TRSM ───────────────────────────────────────────────
    println!("\n--- TRSM (triangular solve) ---");
    let tri = [2.0, 0.0, 3.0, 4.0_f32]; // lower triangular
    let result = trsm(&tri, &[2.0, 11.0], 2, 1, TriangularSide::Lower, DiagonalType::NonUnit)?;
    println!("Lower triangular solve: x = [{:.4}, {:.4}]", result.x[0], result.x[1]);

    // ── BLAS Level-3 ───────────────────────────────────────
    println!("\n--- BLAS Level-3 ---");

    // syrk: C = A·Aᵀ
    let a_syrk = [1.0, 2.0, 3.0, 4.0_f32]; // 2×2
    let mut c_syrk = [0.0_f32; 4];
    syrk(&a_syrk, &mut c_syrk, 2, 2, 1.0, 0.0)?;
    println!("syrk(A·Aᵀ): [{:.1}, {:.1}; {:.1}, {:.1}]",
        c_syrk[0], c_syrk[1], c_syrk[2], c_syrk[3]);

    // syr2k: C = A·Bᵀ + B·Aᵀ
    let b_syr2k = [5.0, 6.0, 7.0, 8.0_f32];
    let mut c_syr2k = [0.0_f32; 4];
    syr2k(&a_syrk, &b_syr2k, &mut c_syr2k, 2, 2, 1.0, 0.0)?;
    println!("syr2k: [{:.1}, {:.1}; {:.1}, {:.1}]",
        c_syr2k[0], c_syr2k[1], c_syr2k[2], c_syr2k[3]);

    // trmm: B = A·B (lower triangular)
    let a_tri = [2.0, 0.0, 3.0, 4.0_f32];
    let mut b_trmm = [1.0, 1.0_f32]; // 2×1
    trmm(&a_tri, &mut b_trmm, 2, 1, 1.0)?;
    println!("trmm(lower·[1,1]ᵀ): [{:.1}, {:.1}]", b_trmm[0], b_trmm[1]);

    // symm: C = A·B (symmetric A)
    let a_sym = [1.0, 2.0, 2.0, 3.0_f32];
    let b_sym = [1.0, 0.0, 0.0, 1.0_f32];
    let mut c_sym = [0.0_f32; 4];
    symm(&a_sym, &b_sym, &mut c_sym, 2, 2, 1.0, 0.0)?;
    println!("symm: [{:.1}, {:.1}; {:.1}, {:.1}]",
        c_sym[0], c_sym[1], c_sym[2], c_sym[3]);

    println!("\n=== All solver demos passed ===");
    Ok(())
}
