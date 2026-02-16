//! GPU non-linear activation benchmarks (sigmoid, tanh, swish, gelu)

use criterion::{BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use trueno::backends::gpu::GpuBackend;

/// Sizes used for all activation benchmarks
const BENCH_SIZES: [usize; 3] = [10_000, 100_000, 1_000_000];

/// Generate narrow-range test data: values in `[-size*0.0005, size*0.0005)` with step 0.001
fn make_narrow_data(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| (i as f32) * 0.001 - (size as f32) * 0.0005)
        .collect()
}

/// Macro to generate a GPU-vs-Scalar activation benchmark.
macro_rules! bench_activation {
    (
        $(#[$meta:meta])*
        fn $fn_name:ident, group = $group:expr,
        data = $data_fn:ident,
        gpu  = |$gpu:ident, $gdata:ident| $gpu_expr:expr,
        scalar = |$sdata:ident| $scalar_expr:expr $(,)?
    ) => {
        $(#[$meta])*
        pub fn $fn_name(c: &mut Criterion) {
            if !GpuBackend::is_available() {
                eprintln!("GPU not available, skipping GPU benchmarks");
                return;
            }

            let mut group = c.benchmark_group($group);

            for size in BENCH_SIZES.iter() {
                group.throughput(Throughput::Elements(*size as u64));

                group.bench_with_input(
                    BenchmarkId::new("GPU", size),
                    size,
                    |bencher, &size| {
                        let $gdata = $data_fn(size);
                        let mut $gpu = GpuBackend::new();

                        bencher.iter(|| {
                            black_box($gpu_expr);
                        });
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("Scalar", size),
                    size,
                    |bencher, &size| {
                        let $sdata = $data_fn(size);

                        bencher.iter(|| {
                            black_box($scalar_expr);
                        });
                    },
                );
            }

            group.finish();
        }
    };
}

bench_activation! {
    /// Benchmark GPU sigmoid activation vs scalar baseline
    fn bench_gpu_sigmoid, group = "gpu_sigmoid",
    data = make_narrow_data,
    gpu    = |gpu, data| gpu.sigmoid(&data).unwrap(),
    scalar = |data| {
        let result: Vec<f32> = data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        result
    },
}

bench_activation! {
    /// Benchmark GPU tanh activation vs scalar baseline
    fn bench_gpu_tanh, group = "gpu_tanh",
    data = make_narrow_data,
    gpu    = |gpu, data| gpu.tanh(&data).unwrap(),
    scalar = |data| {
        let result: Vec<f32> = data.iter().map(|&x| x.tanh()).collect();
        result
    },
}

bench_activation! {
    /// Benchmark GPU swish activation vs scalar baseline
    fn bench_gpu_swish, group = "gpu_swish",
    data = make_narrow_data,
    gpu    = |gpu, data| gpu.swish(&data).unwrap(),
    scalar = |data| {
        let result: Vec<f32> = data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
        result
    },
}

bench_activation! {
    /// Benchmark GPU GELU activation vs scalar baseline
    fn bench_gpu_gelu, group = "gpu_gelu",
    data = make_narrow_data,
    gpu    = |gpu, data| gpu.gelu(&data).unwrap(),
    scalar = |data| {
        const SQRT_2_OVER_PI: f32 = 0.7978846;
        const COEFF: f32 = 0.044715;
        let result: Vec<f32> = data
            .iter()
            .map(|&x| {
                let x_cubed = x * x * x;
                let inner = SQRT_2_OVER_PI * (x + COEFF * x_cubed);
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect();
        result
    },
}
