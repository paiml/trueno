//! Tensor contraction demo — Einstein summation via TTGT.

use trueno_tensor::{einsum, matmul, outer, trace, Tensor};

fn main() {
    println!("=== trueno-tensor: Einstein Summation Demo ===\n");

    // Matrix multiply: ij,jk->ik
    let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let c = matmul(&a, &b).unwrap();
    println!("Matrix multiply (2x3 * 3x2):");
    println!("  C[0,0]={}, C[0,1]={}, C[1,0]={}, C[1,1]={}\n",
        c.get(&[0, 0]), c.get(&[0, 1]), c.get(&[1, 0]), c.get(&[1, 1]));

    // Outer product: i,j->ij
    let u = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let v = Tensor::new(vec![2], vec![4.0, 5.0]).unwrap();
    let op = outer(&u, &v).unwrap();
    println!("Outer product (3 x 2):");
    println!("  shape={:?}, data={:?}\n", op.shape(), op.data());

    // Trace
    let eye = Tensor::new(vec![3, 3], vec![
        1.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 3.0,
    ]).unwrap();
    let tr = trace(&eye).unwrap();
    println!("Trace of diag(1,2,3) = {tr}\n");

    // 3D contraction: ijk,jkl->il
    let t1 = Tensor::new(vec![2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let t2 = Tensor::new(vec![2, 2, 3], (1..=12).map(|i| i as f32).collect()).unwrap();
    let t3 = einsum("ijk,jkl->il", &t1, &t2).unwrap();
    println!("3D contraction ijk,jkl->il:");
    println!("  shape={:?}, data={:?}\n", t3.shape(), t3.data());

    // Transpose
    let m = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mt = m.transpose(&[1, 0]);
    println!("Transpose (2x3 -> 3x2):");
    println!("  shape={:?}, data={:?}", mt.shape(), mt.data());
}
