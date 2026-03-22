#![allow(clippy::disallowed_methods)]
//! # Contract Pipeline Demo
//!
//! Demonstrates the escape-proof contract enforcement pipeline:
//!
//! ```text
//! contracts/*.yaml  →  build.rs reads PRE/POST  →  #[contract] proc macro
//!       ↓                      ↓                          ↓
//!  Lean theorem ref    ENV vars set at build     debug_assert!() injected
//! ```
//!
//! ## How it works
//!
//! 1. `contracts/elementwise-kernel-v1.yaml` defines preconditions:
//!    ```yaml
//!    equations:
//!      relu:
//!        preconditions:
//!          - "!input.is_empty()"
//!          - "output.len() == input.len()"
//!    ```
//!
//! 2. `build.rs` reads the YAML and emits env vars:
//!    ```text
//!    cargo:rustc-env=CONTRACT_ELEMENTWISE_KERNEL_V1_RELU_PRE_0=!input.is_empty()
//!    cargo:rustc-env=CONTRACT_ELEMENTWISE_KERNEL_V1_RELU_PRE_1=output.len() == input.len()
//!    ```
//!
//! 3. `#[contract("elementwise-kernel-v1", equation = "relu")]` reads those env
//!    vars at compile time and injects `debug_assert!()` into the function body.
//!
//! 4. Change the YAML → assertions change automatically at next build.
//!    Remove the YAML → compile error (env var missing).
//!
//! ## Run
//!
//! ```bash
//! cargo run --example contract_pipeline_demo
//! ```

fn main() {
    println!("=== Trueno Contract Pipeline Demo ===\n");

    // Show build-time contract metadata
    println!("Contract binding source: {}", env!("CONTRACT_BINDING_SOURCE"));
    println!("Total bindings: {}", env!("CONTRACT_TOTAL"));
    println!("Implemented: {}", env!("CONTRACT_IMPLEMENTED"));
    println!("Partial: {}", env!("CONTRACT_PARTIAL"));
    println!("Gaps: {}\n", env!("CONTRACT_GAPS"));

    // Demonstrate that contracts are enforced at runtime (debug builds)
    println!("--- ReLU contract (from YAML preconditions) ---");
    let input = vec![1.0f32, -2.0, 3.0, -4.0, 5.0];
    let mut output = vec![0.0f32; 5];
    trueno::blis::elementwise::relu(&input, &mut output).unwrap();
    println!("  relu({:?}) = {:?}", &input, &output);
    println!("  Preconditions checked: !input.is_empty(), output.len() == input.len()");
    println!("  Postconditions checked: output[i] >= 0.0\n");

    // Demonstrate softmax contract
    println!("--- Softmax contract (from YAML preconditions) ---");
    let logits = vec![1.0f32, 2.0, 3.0, 4.0];
    let probs = trueno::blis::softmax::softmax_1d_alloc(&logits);
    let sum: f32 = probs.iter().sum();
    println!("  softmax({:?}) = {:?}", &logits, &probs);
    println!("  sum = {sum:.6} (partition of unity, proven in Lean 4)");
    println!("  Preconditions checked: !logits.is_empty(), logits are finite\n");

    // Demonstrate transpose contract
    println!("--- Transpose contract (from YAML preconditions) ---");
    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let mut b = vec![0.0f32; 6]; // 3x2
    trueno::blis::transpose::transpose(2, 3, &a, &mut b).unwrap();
    println!("  A (2x3) = {:?}", &a);
    println!("  B (3x2) = {:?}", &b);
    println!("  Preconditions checked: rows>0, cols>0, a.len()==rows*cols\n");

    // Demonstrate GEMV contract
    println!("--- GEMV contract (from YAML preconditions) ---");
    let a_vec = vec![1.0f32, 2.0]; // K=2
    let b_mat = vec![1.0f32, 0.0, 0.0, 1.0]; // 2x2 identity
    let mut c = vec![0.0f32; 2]; // N=2
    trueno::blis::gemv::gemv(2, 2, &a_vec, &b_mat, &mut c);
    println!("  gemv([1,2], I_2) = {:?}", &c);
    println!("  Preconditions checked: a.len()>=k, b.len()>=k*n, c.len()>=n\n");

    println!("=== Pipeline Summary ===");
    println!("  YAML contracts:     16 preconditions, 7 postconditions");
    println!("  build.rs env vars:  Set at compile time from YAML");
    println!("  #[contract] macro:  Reads env vars, injects debug_assert!()");
    println!("  Lean theorems:      Referenced in YAML, proven in provable-contracts/lean/");
    println!("  Runtime cost:       Zero in release builds (debug_assert!)");
    println!("\n  Change YAML → assertions change automatically.");
    println!("  Remove YAML → compile_error!()");
    println!("  Remove macro → pmat comply FAILS (CB-1203)");
}
