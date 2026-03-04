//! RNG demonstration: Philox 4x32-10 uniform and normal generation.
//!
//! ```sh
//! cargo run --example rng_demo -p trueno-rand
//! ```

use trueno_rand::Philox4x32;

fn main() {
    println!("=== trueno-rand: Philox 4x32-10 Demo ===\n");

    // 1. Basic generation
    let mut rng = Philox4x32::new(42);
    let vals = rng.next_4u32();
    println!("4 u32 values: {vals:?}");

    // 2. Determinism
    let mut rng2 = Philox4x32::new(42);
    assert_eq!(rng2.next_4u32(), vals);
    println!("Determinism verified: same seed → same output");

    // 3. Uniform distribution
    let mut rng3 = Philox4x32::new(123);
    let mut uniform = vec![0.0f32; 10000];
    rng3.fill_uniform(&mut uniform);
    let mean: f32 = uniform.iter().sum::<f32>() / uniform.len() as f32;
    println!("\nUniform[0,1) — 10k samples:");
    println!("  Mean: {mean:.4} (expected: 0.5)");

    let var: f64 = uniform
        .iter()
        .map(|&v| {
            let d = f64::from(v) - f64::from(mean);
            d * d
        })
        .sum::<f64>()
        / uniform.len() as f64;
    println!("  Variance: {var:.4} (expected: 0.0833)");

    // 4. Normal distribution
    let mut rng4 = Philox4x32::new(456);
    let mut normal = vec![0.0f32; 10000];
    rng4.fill_normal(&mut normal);
    let n_mean: f64 = normal.iter().map(|&v| f64::from(v)).sum::<f64>() / normal.len() as f64;
    let n_var: f64 = normal
        .iter()
        .map(|&v| {
            let d = f64::from(v) - n_mean;
            d * d
        })
        .sum::<f64>()
        / normal.len() as f64;
    println!("\nNormal(0,1) — 10k samples:");
    println!("  Mean: {n_mean:.4} (expected: 0.0)");
    println!("  Variance: {n_var:.4} (expected: 1.0)");

    // 5. Parallel-friendly stateless generation
    let key = [42u32, 0];
    let result = Philox4x32::generate_at(key, [0, 0, 0, 0]);
    println!("\nStateless: generate_at(key, counter) = {result:?}");

    println!("\n=== All RNG demos passed ===");
}
