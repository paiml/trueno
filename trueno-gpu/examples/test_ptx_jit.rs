use trueno_gpu::kernels::{Kernel, ElementwiseMulKernel, GemmKernel};
use trueno_gpu::kernels::optimizer::AdamWStepKernel;

fn main() {
    // elementwise_mul
    let k = ElementwiseMulKernel::new(1024);
    let ptx = k.emit_ptx_for_target("sm_89");
    std::fs::write("/tmp/elementwise_mul_sm89.ptx", &ptx).unwrap();
    println!("elementwise_mul: {} bytes", ptx.len());

    // gemm_tiled (896x896x896 tile=16)
    let k = GemmKernel::tiled(896, 896, 896, 16);
    let ptx = k.emit_ptx_for_target("sm_89");
    std::fs::write("/tmp/gemm_tiled_sm89.ptx", &ptx).unwrap();
    println!("gemm_tiled: {} bytes", ptx.len());

    // adamw_step
    let k = AdamWStepKernel::new(1024);
    let ptx = k.emit_ptx_for_target("sm_89");
    std::fs::write("/tmp/adamw_step_sm89.ptx", &ptx).unwrap();
    println!("adamw_step: {} bytes", ptx.len());
}
