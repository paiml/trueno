//! Bias + Activation Kernel Example
//!
//! This example demonstrates generating a fused bias addition + activation
//! kernel in PTX using trueno-gpu.
//!
//! Run with: `cargo run -p trueno-gpu --example bias_activation`

use trueno_gpu::kernels::{Activation, BiasActivationKernel, Kernel};

fn main() {
    println!("=== trueno-gpu: Bias + Activation Kernel Generation ===\n");

    // Create kernels for different activation functions
    let _no_activation = BiasActivationKernel::new(4096, 256);
    let relu_kernel = BiasActivationKernel::new(4096, 256).with_relu();
    let gelu_kernel = BiasActivationKernel::new(4096, 256).with_gelu();

    println!("Available activation variants:");
    println!("  1. None  - Bias addition only: output[i] += bias[i % bias_size]");
    println!("  2. ReLU  - max(0, x) after bias");
    println!("  3. GELU  - x * sigmoid(1.702 * x) approximation after bias");

    println!("\n--- Bias + ReLU PTX Generation ---\n");

    // Generate PTX for ReLU kernel
    let ptx = relu_kernel.emit_ptx();

    // Print the PTX
    let lines: Vec<&str> = ptx.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        println!("{:4}: {}", i + 1, line);
    }

    println!("\n--- Kernel Details ---");
    println!("Total elements: 4096");
    println!("Bias vector size: 256 (baked into kernel for efficiency)");
    println!("PTX size: {} bytes", ptx.len());
    println!("PTX lines: {}", lines.len());

    println!("\n--- GELU Kernel PTX ---\n");

    // Generate PTX for GELU kernel
    let gelu_ptx = gelu_kernel.emit_ptx();
    let gelu_lines: Vec<&str> = gelu_ptx.lines().collect();
    for (i, line) in gelu_lines.iter().enumerate() {
        println!("{:4}: {}", i + 1, line);
    }

    println!("\n--- GELU Kernel Details ---");
    println!("GELU approximation: x * sigmoid(1.702 * x)");
    println!("Uses ex2.approx for fast exp(-1.702*x)");
    println!("Uses div.rn for reciprocal in sigmoid");
    println!("PTX size: {} bytes", gelu_ptx.len());
    println!("PTX lines: {}", gelu_lines.len());

    // Demonstrate builder pattern
    println!("\n--- Builder Pattern Demo ---");
    let custom = BiasActivationKernel::new(8192, 512).with_activation(Activation::GELU);
    println!("Custom kernel: n=8192, bias_size=512, activation=GELU");
    println!("Kernel name: {}", custom.name());
}
