//! Basic element-wise operations: matmul, add, mul, sub, scale,
//! dot product, activations, clip, and 2D convolution.

/// Matrix multiplication compute shader (WGSL) — tiled shared memory
///
/// Computes C = A × B where:
/// - A is M×K
/// - B is K×N
/// - C is M×N
///
/// Uses 16×16 shared memory tiles to reduce global memory bandwidth by ~16×.
/// Each workgroup loads tiles of A and B into `var<workgroup>` memory, then
/// computes partial products from shared memory.  This is the standard tiled
/// matmul from GPU computing textbooks (KAIZEN-021).
///
/// # Contract (C-TILED-MATMUL-001)
///
/// - **Binding layout**: identical to the naive shader (0=a, 1=b, 2=c, 3=dims)
/// - **Workgroup size**: 16×16 = 256 threads (unchanged)
/// - **Dispatch**: ceil(M/16) × ceil(N/16) workgroups (unchanged)
/// - **Result**: bit-identical to naive shader for all M, K, N (f32 associativity aside)
/// - **Speedup**: 5–15× on real GPUs (bandwidth-bound → compute-bound)
pub(crate) const MATMUL_SHADER: &str = r#"
const TILE: u32 = 16u;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Dimensions {
    M: u32,  // rows of A and C
    K: u32,  // cols of A, rows of B
    N: u32,  // cols of B and C
}

@group(0) @binding(3) var<uniform> dims: Dimensions;

// Shared memory tiles — each 16×16 = 256 floats
var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

// Workgroup size: 16×16 = 256 threads
@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row = global_id.x;
    let col = global_id.y;
    let lr = local_id.x;  // local row within tile [0..15]
    let lc = local_id.y;  // local col within tile [0..15]

    var sum: f32 = 0.0;

    // Iterate over K dimension in tiles of 16
    let num_tiles = (dims.K + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load A tile: A[row, t*TILE + lc]
        let a_col = t * TILE + lc;
        if (row < dims.M && a_col < dims.K) {
            tile_a[lr * TILE + lc] = a[row * dims.K + a_col];
        } else {
            tile_a[lr * TILE + lc] = 0.0;
        }

        // Load B tile: B[t*TILE + lr, col]
        let b_row = t * TILE + lr;
        if (b_row < dims.K && col < dims.N) {
            tile_b[lr * TILE + lc] = b[b_row * dims.N + col];
        } else {
            tile_b[lr * TILE + lc] = 0.0;
        }

        // Wait for all threads to finish loading
        workgroupBarrier();

        // Accumulate partial dot product from shared memory
        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            sum = sum + tile_a[lr * TILE + k] * tile_b[k * TILE + lc];
        }

        // Wait before loading next tile (prevents overwriting while others read)
        workgroupBarrier();
    }

    // Write result
    if (row < dims.M && col < dims.N) {
        c[row * dims.N + col] = sum;
    }
}
"#;

/// PMAT-326: GEMV compute shader (WGSL) — matrix-vector product y = W × x
///
/// Optimized for M=1 (single-token decode). Each workgroup computes ONE output
/// element by cooperatively reducing the dot product along K using shared memory.
///
/// - W: [N, K] row-major weight matrix
/// - x: [K] input vector
/// - y: [N] output vector
///
/// Workgroup: 256 threads. Each workgroup handles 1 output row.
/// Dispatch: N workgroups (one per output element).
/// Reduction: tree reduction in shared memory (log2(256) = 8 steps).
/// PMAT-331: vec4 vectorized GEMV — 4x fewer memory transactions.
/// Each thread loads vec4<f32> (4 floats per load), dot4 in registers.
/// K must be divisible by 4 (true for all Qwen dimensions: 1536, 256, 8960).
pub(crate) const GEMV_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;     // input [K/4]
@group(0) @binding(1) var<storage, read> w: array<vec4<f32>>;     // weight [N, K/4]
@group(0) @binding(2) var<storage, read_write> y: array<f32>;     // output [N]

struct Params {
    n: u32,  // output dim (number of rows)
    k: u32,  // input dim (K, NOT K/4 — shader divides internally)
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg_id.x;
    let tid = lid.x;
    let k4 = params.k / 4u;  // Number of vec4 elements per row

    if (row >= params.n) { return; }

    // Phase 1: vec4 dot product — 4 FMAs per iteration
    var partial_sum: f32 = 0.0;
    let row_offset = row * k4;
    var col4 = tid;
    while (col4 < k4) {
        let wv = w[row_offset + col4];
        let xv = x[col4];
        partial_sum += dot(wv, xv);  // vec4 dot = 4 FMAs
        col4 += 256u;
    }
    sdata[tid] = partial_sum;
    workgroupBarrier();

    // Phase 2: Tree reduction (256 → 1)
    if (tid < 128u) { sdata[tid] += sdata[tid + 128u]; }
    workgroupBarrier();
    if (tid < 64u) { sdata[tid] += sdata[tid + 64u]; }
    workgroupBarrier();
    if (tid < 32u) { sdata[tid] += sdata[tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u) { sdata[tid] += sdata[tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { sdata[tid] += sdata[tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { sdata[tid] += sdata[tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { sdata[tid] += sdata[tid + 2u]; }
    workgroupBarrier();
    if (tid == 0u) {
        y[row] = sdata[0] + sdata[1];
    }
}
"#;

/// Vector addition compute shader (WGSL)
///
/// Computes c = a + b element-wise
pub(crate) const VEC_ADD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&a);

    if (idx < len) {
        c[idx] = a[idx] + b[idx];
    }
}
"#;

/// Element-wise multiplication shader (WGSL)
///
/// Computes c = a * b element-wise
pub(crate) const VEC_MUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&a);

    if (idx < len) {
        c[idx] = a[idx] * b[idx];
    }
}
"#;

/// Element-wise subtraction shader (WGSL)
///
/// Computes c = a - b element-wise
pub(crate) const VEC_SUB_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&a);

    if (idx < len) {
        c[idx] = a[idx] - b[idx];
    }
}
"#;

/// Scalar multiplication shader (WGSL)
///
/// Computes output = input * scalar element-wise
pub(crate) const SCALE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct ScaleParams {
    scalar: f32,
}

@group(0) @binding(2) var<uniform> params: ScaleParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        output[idx] = input[idx] * params.scalar;
    }
}
"#;

/// Dot product reduction shader (WGSL)
///
/// Computes sum(a[i] * b[i]) using parallel reduction
pub(crate) const DOT_PRODUCT_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

var<workgroup> partial_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let idx = global_id.x;
    let local_idx = local_id.x;
    let len = arrayLength(&a);

    // Load and multiply
    var sum: f32 = 0.0;
    if (idx < len) {
        sum = a[idx] * b[idx];
    }
    partial_sums[local_idx] = sum;

    workgroupBarrier();

    // Parallel reduction within workgroup
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_idx < stride) {
            partial_sums[local_idx] = partial_sums[local_idx] + partial_sums[local_idx + stride];
        }
        stride = stride / 2u;
        workgroupBarrier();
    }

    // First thread writes workgroup result
    if (local_idx == 0u) {
        result[global_id.x / 256u] = partial_sums[0];
    }
}
"#;

/// ReLU activation compute shader (WGSL)
///
/// Computes element-wise ReLU: max(0, x)
///
/// This is one of the simplest GPU operations - a single comparison and selection per element.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub(crate) const RELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        // ReLU: max(0, x)
        output[idx] = max(0.0, input[idx]);
    }
}
"#;

/// Leaky ReLU activation compute shader (WGSL)
///
/// Computes element-wise Leaky ReLU: leaky_relu(x, α) = max(αx, x) = x if x > 0, else αx
///
/// Leaky ReLU addresses the "dying ReLU" problem by allowing small negative activations.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub(crate) const LEAKY_RELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct LeakyReluParams {
    negative_slope: f32,
}

@group(0) @binding(2) var<uniform> params: LeakyReluParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        let x = input[idx];

        // Leaky ReLU: leaky_relu(x, α) = x if x > 0, else αx
        if (x > 0.0) {
            output[idx] = x;
        } else {
            output[idx] = params.negative_slope * x;
        }
    }
}
"#;

/// ELU (Exponential Linear Unit) activation compute shader (WGSL)
///
/// Computes element-wise ELU: elu(x, α) = x if x > 0, else α(e^x - 1)
///
/// ELU has smooth gradients everywhere and pushes mean activations closer to zero,
/// improving learning in deep networks.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub(crate) const ELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct EluParams {
    alpha: f32,
}

@group(0) @binding(2) var<uniform> params: EluParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        let x = input[idx];

        // ELU: elu(x, α) = x if x > 0, else α(e^x - 1)
        if (x > 0.0) {
            output[idx] = x;
        } else {
            output[idx] = params.alpha * (exp(x) - 1.0);
        }
    }
}
"#;

/// Sigmoid activation compute shader (WGSL)
///
/// Computes element-wise sigmoid: σ(x) = 1 / (1 + e^(-x))
///
/// Classic logistic function used in binary classification and attention mechanisms.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub(crate) const SIGMOID_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        let x = input[idx];

        // Sigmoid: σ(x) = 1 / (1 + exp(-x))
        // Numerically stable implementation:
        // For x >= 0: σ(x) = 1 / (1 + exp(-x))
        // For x < 0: σ(x) = exp(x) / (1 + exp(x))
        var result: f32;
        if (x >= 0.0) {
            result = 1.0 / (1.0 + exp(-x));
        } else {
            let exp_x = exp(x);
            result = exp_x / (1.0 + exp_x);
        }

        output[idx] = result;
    }
}
"#;

/// Tanh (hyperbolic tangent) activation compute shader (WGSL)
///
/// Computes element-wise tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
///
/// Classic activation function used in LSTM, GRU, and traditional neural networks.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub(crate) const TANH_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        let x = input[idx];

        // Tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        //                = (e^(2x) - 1) / (e^(2x) + 1)
        // Numerically stable implementation:
        // For |x| > 20: tanh(x) ≈ sign(x) (saturates at ±1)
        // Otherwise: use standard formula
        var result: f32;
        if (x > 20.0) {
            result = 1.0;
        } else if (x < -20.0) {
            result = -1.0;
        } else {
            let exp_2x = exp(2.0 * x);
            result = (exp_2x - 1.0) / (exp_2x + 1.0);
        }

        output[idx] = result;
    }
}
"#;

/// Swish activation compute shader (WGSL)
///
/// Computes element-wise swish: swish(x) = x * σ(x) = x / (1 + e^(-x))
///
/// Modern activation function (SiLU) used in transformers and modern architectures.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub(crate) const SWISH_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        let x = input[idx];

        // Swish: swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
        // Numerically stable implementation:
        // For x >= 0: swish(x) = x / (1 + exp(-x))
        // For x < 0: swish(x) = x * exp(x) / (1 + exp(x))
        var result: f32;
        if (x >= 0.0) {
            result = x / (1.0 + exp(-x));
        } else {
            let exp_x = exp(x);
            result = x * exp_x / (1.0 + exp_x);
        }

        output[idx] = result;
    }
}
"#;

/// GELU activation compute shader (WGSL)
///
/// Computes element-wise GELU using tanh approximation:
/// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
///
/// Standard activation in BERT, GPT-2, GPT-3, and modern transformers.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub(crate) const GELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        let x = input[idx];

        // GELU approximation (tanh-based):
        // GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        let SQRT_2_OVER_PI: f32 = 0.7978846; // √(2/π)
        let COEFF: f32 = 0.044715;

        let x_cubed = x * x * x;
        let inner = SQRT_2_OVER_PI * (x + COEFF * x_cubed);
        let result = 0.5 * x * (1.0 + tanh(inner));

        output[idx] = result;
    }
}
"#;

/// Clip (clamp) compute shader (WGSL)
///
/// Computes element-wise clip: clamp(x, min_val, max_val)
///
/// Constrains values to the range [min_val, max_val].
/// GPU acceleration beneficial for large vectors (>100K elements).
pub(crate) const CLIP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct ClipParams {
    min_val: f32,
    max_val: f32,
}

@group(0) @binding(2) var<uniform> params: ClipParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        // Clip: clamp(x, min_val, max_val) = max(min_val, min(max_val, x))
        output[idx] = clamp(input[idx], params.min_val, params.max_val);
    }
}
"#;

/// 2D Convolution compute shader (WGSL)
///
/// Computes 2D convolution: output = input ⊗ kernel
/// Uses "valid" padding (no padding, output smaller than input)
///
/// Output dimensions:
/// - output_rows = input_rows - kernel_rows + 1
/// - output_cols = input_cols - kernel_cols + 1
///
/// Uses workgroups of 16×16 threads for optimal GPU utilization
pub(crate) const CONVOLVE2D_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> kernel: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct ConvDimensions {
    input_rows: u32,
    input_cols: u32,
    kernel_rows: u32,
    kernel_cols: u32,
    output_rows: u32,
    output_cols: u32,
}

@group(0) @binding(3) var<uniform> dims: ConvDimensions;

// Workgroup size: 16×16 = 256 threads
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_row = global_id.x;
    let out_col = global_id.y;

    // Bounds check
    if (out_row >= dims.output_rows || out_col >= dims.output_cols) {
        return;
    }

    var sum: f32 = 0.0;

    // Apply kernel: iterate over kernel dimensions
    for (var k_row: u32 = 0u; k_row < dims.kernel_rows; k_row = k_row + 1u) {
        for (var k_col: u32 = 0u; k_col < dims.kernel_cols; k_col = k_col + 1u) {
            // Input pixel coordinates
            let in_row = out_row + k_row;
            let in_col = out_col + k_col;

            // Input and kernel are row-major
            let input_idx = in_row * dims.input_cols + in_col;
            let kernel_idx = k_row * dims.kernel_cols + k_col;

            sum = sum + input[input_idx] * kernel[kernel_idx];
        }
    }

    // Write output (row-major)
    let output_idx = out_row * dims.output_cols + out_col;
    output[output_idx] = sum;
}
"#;
