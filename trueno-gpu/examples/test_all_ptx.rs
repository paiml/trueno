use trueno_gpu::kernels::{Kernel, Q5_0GemvKernel, RmsNormKernel, SoftmaxKernel};

fn main() {
    // Test RMSNorm kernel
    println!("=== RMSNorm Kernel ===");
    let rms = RmsNormKernel::new(896);
    let ptx = rms.emit_ptx();
    std::fs::write("/tmp/rmsnorm.ptx", &ptx).unwrap();
    println!(
        "RMSNorm PTX written to /tmp/rmsnorm.ptx ({} bytes)",
        ptx.len()
    );

    // Test Q5_0 kernel (this is the one failing)
    println!("=== Q5_0 GEMV Kernel ===");
    let q5 = Q5_0GemvKernel::new(896, 896);
    let ptx = q5.emit_ptx();
    std::fs::write("/tmp/q5_gemv.ptx", &ptx).unwrap();
    println!("Q5_0 PTX written to /tmp/q5_gemv.ptx ({} bytes)", ptx.len());

    // Test Softmax kernel
    println!("=== Softmax Kernel ===");
    let sm = SoftmaxKernel::new(896);
    let ptx = sm.emit_ptx();
    std::fs::write("/tmp/softmax.ptx", &ptx).unwrap();
    println!(
        "Softmax PTX written to /tmp/softmax.ptx ({} bytes)",
        ptx.len()
    );
}
