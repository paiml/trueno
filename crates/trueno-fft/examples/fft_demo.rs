//! FFT demonstration: forward, inverse, Parseval verification, R2C.
//!
//! ```sh
//! cargo run --example fft_demo -p trueno-fft
//! ```

use trueno_fft::{Complex, FftPlan};

fn main() {
    println!("=== trueno-fft: Stockham FFT Demo ===\n");

    // 1. Basic forward FFT
    let plan = FftPlan::new(8).expect("valid plan");
    let input: Vec<Complex> = (0..8).map(|i| Complex::new(i as f32, 0.0)).collect();
    let mut output = vec![Complex::ZERO; 8];

    plan.forward(&input, &mut output).expect("forward FFT");
    println!("Forward FFT of [0, 1, 2, ..., 7]:");
    for (k, x) in output.iter().enumerate() {
        println!("  X[{k}] = {:.4} + {:.4}i", x.re, x.im);
    }

    // 2. Parseval energy conservation
    let energy_time: f32 = input.iter().map(|x| x.norm_sq()).sum();
    let energy_freq: f32 = output.iter().map(|x| x.norm_sq()).sum::<f32>() / 8.0;
    println!("\nParseval check:");
    println!("  Time-domain energy:  {energy_time:.6}");
    println!("  Freq-domain energy:  {energy_freq:.6}");
    println!(
        "  Error: {:.2e} {}",
        (energy_time - energy_freq).abs(),
        if (energy_time - energy_freq).abs() < 1e-3 {
            "✓"
        } else {
            "✗"
        }
    );

    // 3. Inverse roundtrip
    let mut recovered = vec![Complex::ZERO; 8];
    plan.inverse(&output, &mut recovered).expect("inverse FFT");
    let max_err: f32 = input
        .iter()
        .zip(recovered.iter())
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0f32, f32::max);
    println!("\nInverse roundtrip max error: {max_err:.2e}");

    // 4. R2C
    let real_input: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let mut r2c_output = vec![Complex::ZERO; 5]; // N/2+1
    plan.forward_r2c(&real_input, &mut r2c_output)
        .expect("R2C");
    println!("\nR2C FFT of [0..8] (N/2+1 = 5 complex outputs):");
    for (k, x) in r2c_output.iter().enumerate() {
        println!("  X[{k}] = {:.4} + {:.4}i", x.re, x.im);
    }

    // 5. Impulse test
    let mut impulse = vec![Complex::ZERO; 8];
    impulse[0] = Complex::new(1.0, 0.0);
    let mut impulse_out = vec![Complex::ZERO; 8];
    plan.forward(&impulse, &mut impulse_out)
        .expect("impulse FFT");
    let impulse_err: f32 = impulse_out
        .iter()
        .map(|x| (x.re - 1.0).abs() + x.im.abs())
        .fold(0.0f32, f32::max);
    println!("\nImpulse response max error: {impulse_err:.2e}");

    println!("\n=== All FFT demos passed ===");
}
