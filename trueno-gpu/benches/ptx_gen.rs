//! PTX Generation Benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use trueno_gpu::ptx::{PtxModule, PtxKernel, PtxType};

fn bench_module_emit(c: &mut Criterion) {
    let kernel = PtxKernel::new("vector_add")
        .param(PtxType::U64, "a_ptr")
        .param(PtxType::U64, "b_ptr")
        .param(PtxType::U64, "c_ptr")
        .param(PtxType::U32, "n");

    let module = PtxModule::new()
        .version(8, 0)
        .target("sm_70")
        .address_size(64)
        .add_kernel(kernel);

    c.bench_function("ptx_module_emit", |b| {
        b.iter(|| {
            black_box(module.emit())
        })
    });
}

criterion_group!(benches, bench_module_emit);
criterion_main!(benches);
