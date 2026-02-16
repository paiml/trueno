//! GPU activation function benchmarks (relu, leaky_relu, elu, clip, sigmoid, tanh, swish, gelu)

use criterion::{BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use trueno::backends::gpu::GpuBackend;

/// Sizes used for all activation benchmarks
const BENCH_SIZES: [usize; 3] = [10_000, 100_000, 1_000_000];

/// Generate centered test data: values in `[-size*0.25, size*0.25)` with step 0.5
fn make_centered_data(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| (i as f32) * 0.5 - (size as f32) * 0.25)
        .collect()
}

/// Generate narrow-range test data: values in `[-size*0.0005, size*0.0005)` with step 0.001
fn make_narrow_data(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| (i as f32) * 0.001 - (size as f32) * 0.0005)
        .collect()
}

/// Generate positive-only test data: values in `[0, size*0.5)` with step 0.5
fn make_positive_data(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) * 0.5).collect()
}

/// Macro to generate a GPU-vs-Scalar activation benchmark.
///
/// Eliminates the repeated DataTransformation boilerplate: GPU availability
/// guard, benchmark group creation, size iteration, data generation,
/// GPU bench arm, scalar bench arm, and group finish.
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
    /// Benchmark GPU ReLU activation vs scalar baseline
    ///
    /// Tests GPU acceleration for element-wise operations.
    /// GPU threshold: >100K elements (OpComplexity::Low)
    fn bench_gpu_relu, group = "gpu_relu",
    data = make_centered_data,
    gpu    = |gpu, data| gpu.relu(&data).unwrap(),
    scalar = |data| {
        let result: Vec<f32> = data.iter().map(|&x| x.max(0.0)).collect();
        result
    },
}

bench_activation! {
    /// Benchmark GPU leaky ReLU activation vs scalar baseline
    fn bench_gpu_leaky_relu, group = "gpu_leaky_relu",
    data = make_centered_data,
    gpu    = |gpu, data| gpu.leaky_relu(&data, 0.01).unwrap(),
    scalar = |data| {
        let negative_slope = 0.01_f32;
        let result: Vec<f32> = data
            .iter()
            .map(|&x| if x > 0.0 { x } else { negative_slope * x })
            .collect();
        result
    },
}

bench_activation! {
    /// Benchmark GPU ELU activation vs scalar baseline
    fn bench_gpu_elu, group = "gpu_elu",
    data = make_centered_data,
    gpu    = |gpu, data| gpu.elu(&data, 1.0).unwrap(),
    scalar = |data| {
        let alpha = 1.0_f32;
        let result: Vec<f32> = data
            .iter()
            .map(|&x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
            .collect();
        result
    },
}

bench_activation! {
    /// Benchmark GPU clip operation vs scalar baseline
    fn bench_gpu_clip, group = "gpu_clip",
    data = make_positive_data,
    gpu    = |gpu, data| gpu.clip(&data, 100.0, 5000.0).unwrap(),
    scalar = |data| {
        let (min_val, max_val) = (100.0_f32, 5000.0_f32);
        let result: Vec<f32> = data.iter().map(|&x| x.max(min_val).min(max_val)).collect();
        result
    },
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
