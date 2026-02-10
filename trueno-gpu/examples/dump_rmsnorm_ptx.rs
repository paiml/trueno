use trueno_gpu::kernels::{BatchedVectorizedRmsNormKernel, Kernel, VectorizedRmsNormKernel};

fn main() {
    let hidden_size = 1536; // Qwen2.5-1.5B hidden_dim

    // Single-vector RMSNorm
    let single = VectorizedRmsNormKernel::new(hidden_size);
    let single_ptx = single.emit_ptx();

    // Batched RMSNorm with M=1 (should match single)
    let batched = BatchedVectorizedRmsNormKernel::new(hidden_size, 1);
    let batched_ptx = batched.emit_ptx();

    eprintln!("=== SINGLE-VECTOR RMSNorm (hidden={}) ===", hidden_size);
    eprintln!("{}", single_ptx);
    eprintln!("\n\n=== BATCHED RMSNorm (hidden={}, M=1) ===", hidden_size);
    eprintln!("{}", batched_ptx);
}
