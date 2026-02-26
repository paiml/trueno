//! Dump CoalescedQ6K PTX for inspection
use trueno_gpu::kernels::{CoalescedQ6KGemvKernel, Kernel};
use trueno_gpu::ptx::PtxModule;

fn main() {
    // Create kernel for typical dimensions
    let kernel = CoalescedQ6KGemvKernel::new(1536, 8960);
    let ptx_kernel = kernel.build_ptx();

    let module =
        PtxModule::new().version(8, 0).target("sm_86").address_size(64).add_kernel(ptx_kernel);

    let ptx = module.emit();
    println!("{}", ptx);
}
