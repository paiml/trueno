use trueno_gpu::kernels::optimizer::AdamWStepKernel;
use trueno_gpu::kernels::{ElementwiseMulKernel, GemmKernel, Kernel};

fn main() {
    // elementwise_mul
    let k = ElementwiseMulKernel::new(1024);
    let ptx = k.emit_ptx_for_target("sm_89");
    std::fs::write("/tmp/elementwise_mul_sm89.ptx", &ptx)
        .expect("failed to write elementwise_mul PTX");
    println!("elementwise_mul: {} bytes", ptx.len());

    // gemm_tiled (896x896x896 tile=16)
    let k = GemmKernel::tiled(896, 896, 896, 16);
    let ptx = k.emit_ptx_for_target("sm_89");
    std::fs::write("/tmp/gemm_tiled_sm89.ptx", &ptx).expect("failed to write gemm_tiled PTX");
    println!("gemm_tiled: {} bytes", ptx.len());

    // adamw_step
    let k = AdamWStepKernel::new(1024);
    let ptx = k.emit_ptx_for_target("sm_89");
    std::fs::write("/tmp/adamw_step_sm89.ptx", &ptx).expect("failed to write adamw_step PTX");
    println!("adamw_step: {} bytes", ptx.len());
}
